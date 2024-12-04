import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import GATConv

import os
import argparse
import time
from datetime import datetime
import random
from typing import Union, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.stats import pearsonr


class SAGEConvCat(MessagePassing):
    """
    *Note: Source function taken from PyTorch Geometric and modified such that
    embeddings are first concatenated and then reduced to out_channel size as
    per the original GraphSAGE paper.
    
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    PyTorch Geometric citation:
    @inproceedings{Fey/Lenssen/2019,
      title={Fast Graph Representation Learning with {PyTorch Geometric}},
      author={Fey, Matthias and Lenssen, Jan E.},
      booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
      year={2019},
    }
    """
    
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEConvCat, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = nn.Linear(in_channels[0]*2, out_channels, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        
        out = self.propagate(edge_index, x=x, size=size)

        ### Concatenation
        out = torch.cat([x, out], dim=-1)
        out = self.lin_l(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GCN_regression(nn.Module):
    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_layer_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines regression model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Size of predicted output tensor for batch size of N, 
            i.e. N x num_classes(=1)

        Returns
        -------
        None.

        '''
        
        super(GCN_regression, self).__init__()
        
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout = 0.5
    
        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
            self.conv3 = SAGEConvCat(graph_conv_layer_sizes[2], graph_conv_layer_sizes[3])
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
        
        self.loss_calc = nn.MSELoss()

        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Predicted expression levels

        '''
        ### Graph convolution module
        if self.num_graph_conv_layers == 1:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 2:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 3:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
            h = self.conv3(h, edge_index)
            h = torch.relu(h)

        h = F.dropout(h, p = self.dropout, training=train_status)

        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        if len(scores.size()) > 1:
            scores = scores.squeeze()
            
        return scores
    
    
    def loss(self, scores, targets):
        '''
        Calculates mean squared error loss
        
        Parameters
        ----------
        scores [tensor]: Predicted scores from forward function
        labels [tensor]: Target scores 

        Returns
        -------
        mse [tensor]: Mean squared error loss

        '''
        
        mse = self.loss_calc(scores, targets)

        return mse

    def to_cpu_npy(x):
        '''
        Simple helper function to transfer GPU tensors to CPU numpy matrices
    
        Parameters
        ----------
        x [tensor]: PyTorch tensor stored on GPU
    
        Returns
        -------
        new_x [array]: Numpy array stored on CPU
    
        '''
    
        new_x = x.cpu().detach().numpy()
        
        return new_x
        
class GCN_classification(nn.Module):

    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_layer_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines classification model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted(=2)

        Returns
        -------
        None.

        '''
        
        super(GCN_classification, self).__init__()

        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout_value = 0.5

        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
            self.conv3 = SAGEConvCat(graph_conv_layer_sizes[2], graph_conv_layer_sizes[3])
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
            
        self.loss_calc = nn.CrossEntropyLoss()
        self.torch_softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Pre-normalized class scores

        '''

        ### Graph convolution module
        if self.num_graph_conv_layers == 1:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 2:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 3:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
            h = self.conv3(h, edge_index)
            h = torch.relu(h)
            
        h = F.dropout(h, p = self.dropout_value, training=train_status)

        ### Linear module
        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        return scores
    
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores from forward function
        labels [tensor]: Class labels for nodes

        Returns
        -------
        xent_loss [tensor]: Cross-entropy loss

        '''

        xent_loss = self.loss_calc(scores, labels)

        return xent_loss
    
    
    def calc_softmax_pred(self, scores):
        '''
        Calculates softmax scores and predicted classes

        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores

        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class

        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted


class SAGEConvCat_weighted(MessagePassing):
    """
    *Note: Source function taken from PyTorch Geometric and modified such that
    embeddings are first concatenated and then reduced to out_channel size as
    per the original GraphSAGE paper.
    
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    PyTorch Geometric citation:
    @inproceedings{Fey/Lenssen/2019,
      title={Fast Graph Representation Learning with {PyTorch Geometric}},
      author={Fey, Matthias and Lenssen, Jan E.},
      booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
      year={2019},
    }
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, num_nodes: int, edge_attr: Tensor,
                 normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        
        super(SAGEConvCat_weighted, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weights_init = False
        #self.inp_size = 10

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        bias = bool(bias)
        self.lin_l = nn.Linear(in_channels[0]*2, out_channels)
        self.edge_weights = torch.nn.Parameter(edge_attr)
        self.num_nodes = num_nodes
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

        #changed
        if self.weights_init:
            torch.nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                train_status: bool, size: Size = None) -> Tensor:

        self.training_status = train_status
        
        # if self.weights_init == False:
        #     self.edge_weights = torch.nn.Parameter(torch.rand(edge_index.shape[1], dtype=torch.float32))
        #     self.num_nodes = x.shape[0]
        #     self.weights_init = True

        out = self.propagate(edge_index, x=x, size=size)

        ### Concatenation
        out = torch.cat([x, out], dim=-1)
        out = self.lin_l(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        
        #self.edge_weights = F.dropout(self.edge_weights, p=0.1, training=self.training_status)
            
        adj_t = adj_t.set_value(self.edge_weights)
        out = matmul(adj_t, x[0], reduce=self.aggr)
        out = F.leaky_relu(out)
        
        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN_classification_weighted(nn.Module):

    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_layer_sizes, num_lin_layers, lin_hidden_sizes, num_classes, num_nodes, edge_attr):
        '''
        Defines classification model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted(=2)

        Returns
        -------
        None.

        '''
        
        super(GCN_classification_weighted, self).__init__()

        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout_value = 0.5

        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat_weighted(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1], num_nodes, edge_attr)
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat_weighted(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1], num_nodes, edge_attr)
            self.conv2 = SAGEConvCat_weighted(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2], num_nodes, edge_attr)
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat_weighted(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1], num_nodes, edge_attr)
            self.conv2 = SAGEConvCat_weighted(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2], num_nodes, edge_attr)
            self.conv3 = SAGEConvCat_weighted(graph_conv_layer_sizes[2], graph_conv_layer_sizes[3], num_nodes, edge_attr)
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
            
        self.loss_calc = nn.CrossEntropyLoss()
        self.torch_softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Pre-normalized class scores

        '''

        ### Graph convolution module
        if self.num_graph_conv_layers == 1:
            h = self.conv1(x, edge_index, train_status)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 2:
            h = self.conv1(x, edge_index, train_status)
            h = torch.relu(h)
            h = self.conv2(h, edge_index, train_status)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 3:
            h = self.conv1(x, edge_index, train_status)
            h = torch.relu(h)
            h = self.conv2(h, edge_index, train_status)
            h = torch.relu(h)
            h = self.conv3(h, edge_index, train_status)
            h = torch.relu(h)
            
        h = F.dropout(h, p = self.dropout_value, training=train_status)

        ### Linear module
        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        return scores
    
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores from forward function
        labels [tensor]: Class labels for nodes

        Returns
        -------
        xent_loss [tensor]: Cross-entropy loss

        '''

        xent_loss = self.loss_calc(scores, labels)

        return xent_loss
    
    
    def calc_softmax_pred(self, scores):
        '''
        Calculates softmax scores and predicted classes

        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores

        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class

        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted
        
class MLP_classification(nn.Module):

    def __init__(self, num_feat, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines classification model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted(=2)

        Returns
        -------
        None.

        '''
        
        super(MLP_classification, self).__init__()
        
        self.num_lin_layers = num_lin_layers
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
            
        self.loss_calc = nn.CrossEntropyLoss()
        self.torch_softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Pre-normalized class scores

        '''

        ### Linear module
        if self.num_lin_layers == 1:
            x = self.lin1(x)
        elif self.num_lin_layers == 2:
            x = self.lin1(x)
            x = torch.relu(x)
            x = self.lin2(x)
        elif self.num_lin_layers == 3:
            x = self.lin1(x)
            x = torch.relu(x)
            x = self.lin2(x)
            x = torch.relu(x)
            x = self.lin3(x)
        
        return x
    
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores from forward function
        labels [tensor]: Class labels for nodes

        Returns
        -------
        xent_loss [tensor]: Cross-entropy loss

        '''

        xent_loss = self.loss_calc(scores, labels)

        return xent_loss
    
    
    def calc_softmax_pred(self, scores):
        '''
        Calculates softmax scores and predicted classes

        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores

        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class

        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted
    
    
class MLP_regression(nn.Module):

    def __init__(self, num_feat, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines regression model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Size of predicted output tensor for batch size of N, 
            i.e. N x num_classes(=1)

        Returns
        -------
        None.

        '''
        
        super(MLP_regression, self).__init__()
        
        self.num_lin_layers = num_lin_layers
   
        self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
        
        self.loss_calc = nn.MSELoss()

        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Predicted expression levels

        '''

        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        
        if len(x.size()) > 1:
            x = x.squeeze()
            
        return x
    
    
    def loss(self, scores, targets):
        '''
        Calculates mean squared error loss
        
        Parameters
        ----------
        scores [tensor]: Predicted scores from forward function
        labels [tensor]: Target scores 

        Returns
        -------
        mse [tensor]: Mean squared error loss

        '''
        
        mse = self.loss_calc(scores, targets)

        return mse

    
INPUT_LENGTH = 6
NUM_CLASSES = 2 
BATCH_SIZE = 64

class CNN(nn.Module):
    def __init__(self, num_conv_layers, num_linear_layers, dropout_rate=0.2):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = 1
        out_channels = 16
        current_length = INPUT_LENGTH

        for i in range(num_conv_layers):
            conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2) if current_length // 2 > 0 else nn.Identity(),
            )
            self.conv_layers.append(conv)
            
            in_channels = out_channels
            out_channels *= 2
            current_length = max(1, current_length // 2)
        
        self.flatten = nn.Flatten()
        linear_input_size = in_channels * current_length

        self.linear_layers = nn.ModuleList()
        if num_linear_layers > 1:
            for i in range(num_linear_layers - 1):
                next_size = int(linear_input_size // 2)
                fc = nn.Sequential(
                    nn.Linear(int(linear_input_size), next_size),
                    nn.ReLU()
                )
                self.linear_layers.append(fc)
                linear_input_size = next_size

        self.final_linear = nn.Linear(linear_input_size, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x):
        out = self.conv_layers[0](x)
        
        for layer in self.conv_layers[1:]:
            out = layer(out)

        out = self.dropout(out)
        out = self.flatten(out)
        for layer in self.linear_layers:
            out = layer(out)
        
        out = self.final_linear(out)
        
        return out
    
    def calculate_accuracy(self, dataset):
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        num_correct = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in data_loader:
                optimizer.zero_grad()
                
                output = model(batch_inputs)
                pred = torch.argmax(output, dim=1)
                num_correct += torch.sum(pred == batch_labels)
            
        return float(num_correct / len(dataset))

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, dropout):
        super(GAT, self).__init__()
        # First GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels[0], heads=num_heads, concat=True)
        # Second GAT layer
        self.gat2 = GATConv(hidden_channels[0] * num_heads, hidden_channels[1], heads=num_heads, concat=False)
        # Third GAT layer
        # self.gat3 = GATConv(hidden_channels[1] * num_heads, hidden_channels[2], heads=num_heads, concat=False)

        # Fully connected layers
        self.ff1 = nn.Linear(hidden_channels[1], hidden_channels[1] // 2)
        self.ff2 = nn.Linear(hidden_channels[1] // 2, 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Pass through GAT layers
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        # x = torch.relu(self.gat3(x, edge_index, edge_attr))
        # Apply dropout and pass through the fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.ff1(x))
        x = self.ff2(x)
        return x

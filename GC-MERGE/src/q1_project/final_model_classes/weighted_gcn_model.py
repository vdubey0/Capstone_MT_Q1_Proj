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

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        
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
        
        
    def forward(self, data):
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

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

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
            
        h = F.dropout(h, p = self.dropout_value)

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
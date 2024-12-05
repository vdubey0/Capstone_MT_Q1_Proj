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
        
        
    def forward(self, data):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask

        Returns
        -------
        scores [tensor]: Pre-normalized class scores

        '''
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
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
        num_lin_layers [int]: Number of linear layers (1 or 2)
        lin_hidden_sizes [list]: Embedding sizes of linear layers
        num_classes [int]: Size of predicted output tensor for batch size of N, 
            i.e. N x num_classes(=1)

        Returns
        -------
        None.
        '''
        super(MLP_regression, self).__init__()
        
        self.num_lin_layers = num_lin_layers
   
        self.lin1 = nn.Linear(num_feat, lin_hidden_sizes[0])
        self.lin2 = nn.Linear(lin_hidden_sizes[0], num_classes)
        
        self.loss_calc = nn.MSELoss()

    def forward(self, x, edge_index=None, train_status=False):
        '''
        Forward function
        
        Parameters
        ----------
        x [tensor]: Input features
        edge_index [tensor]: Not used in this implementation
        train_status [bool]: optional, not used in this implementation

        Returns
        -------
        x [tensor]: Predicted values
        '''
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        
        if len(x.size()) > 1:
            x = x.squeeze()
            
        return x
    
    def loss(self, scores, targets):
        '''
        Calculates mean squared error loss
        
        Parameters
        ----------
        scores [tensor]: Predicted scores from forward function
        targets [tensor]: Target scores 

        Returns
        -------
        mse [tensor]: Mean squared error loss
        '''
        mse = self.loss_calc(scores, targets)
        return mse
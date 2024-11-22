import numpy as np
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd

import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn

from model_classes_ import GCN_classification, GCN_regression

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
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
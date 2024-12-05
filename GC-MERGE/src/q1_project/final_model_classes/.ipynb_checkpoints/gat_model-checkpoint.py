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

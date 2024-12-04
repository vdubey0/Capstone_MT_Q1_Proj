import os

import numpy as np
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd

import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import GATConv

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("No GPU available")

cell_line = 'E116'
regression_flag = 0
chip_res = 10000
hic_res = 10000
num_hm = 6
num_feat = int((hic_res/chip_res)*num_hm)
num_classes = 2 if regression_flag == 0 else 1

#define data paths
src_dir = os.getcwd()
#src_dir = os.path.dirname(base_path)
save_dir = os.path.join(src_dir, 'data', cell_line, 'saved_runs')
hic_sparse_mat_file = os.path.join(src_dir, 'data', cell_line, 'hic_sparse.npz')
np_nodes_lab_genes_file = os.path.join(src_dir, 'data',  cell_line, \
    'np_nodes_lab_genes_reg' + str(regression_flag) + '.npy')
np_hmods_norm_all_file = os.path.join(src_dir, 'data', cell_line, \
    'np_hmods_norm_chip_' + str(chip_res) + 'bp.npy')

mat = load_npz(hic_sparse_mat_file)
allNodes_hms = np.load(np_hmods_norm_all_file) #contains 6 histone marks for all 279606 regions + id (Shape = [279606, 7])
hms = allNodes_hms[:, 1:] #only includes features, not node ids (Shape = [279606, 6])
X = torch.tensor(hms).float().reshape(-1, num_feat) #convert hms to tensor (Shape = [279606, 6])
allNodes = allNodes_hms[:, 0].astype(int) #contains ids of all regions (Shape = [279606, 1])

geneNodes_labs = np.load(np_nodes_lab_genes_file) #contains the expression level of each gene (Shape = [16699, 2])
geneNodes = geneNodes_labs[:, -2].astype(int) #contains ids of regions that encode a gene (Shape = [16699, 1])

allLabs = -1*np.ones(np.shape(allNodes))
targetNode_mask = torch.tensor(geneNodes).long()
geneLabs = geneNodes_labs[:, -1].astype(int)
allLabs[geneNodes] = geneLabs #contains expression level for each region (-1 if region doesn't encode gene, 1 if gene is expressed, 0 if not)
Y = torch.tensor(allLabs).long()

extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)

data = torch_geometric.data.Data(edge_index = extract[0], edge_attr = extract[1], x = X, y = Y)
G = data

pred_idx_shuff = torch.randperm(targetNode_mask.shape[0])
fin_train = np.floor(0.7*pred_idx_shuff.shape[0]).astype(int)
fin_valid = np.floor(0.85*pred_idx_shuff.shape[0]).astype(int)
train_idx = pred_idx_shuff[:fin_train]
valid_idx = pred_idx_shuff[fin_train:fin_valid]
test_idx = pred_idx_shuff[fin_valid:]

def to_cpu_npy(x):
    return x.cpu().detach().numpy()
    
def train_model_classification(model, loss, graph, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer):
    model = model.to(device)
    graph = graph.to(device)

    optimizer = optimizer
    
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    
    model.train()
    train_status = True
    
    print('\n')

    train_losses = []
    valid_losses = []
    for e in list(range(max_epoch)):
        model.train()
        optimizer.zero_grad()

        all_scores = model(graph)[targetNode_mask]
        train_scores = all_scores[train_idx]
        
        train_loss = loss(train_scores, torch.LongTensor(train_labels).to(device))
        train_losses.append(train_loss.item())

        train_loss.backward()
        optimizer.step()

        model.eval()
        valid_scores = all_scores[valid_idx]
        valid_loss = loss(valid_scores, torch.LongTensor(valid_labels).to(device))
        valid_losses.append(valid_loss.item())

        if e%100 == 0:
            print(f'Epoch {e}: Train Loss = {train_loss}, Valid Loss = {valid_loss}')

    return train_losses, valid_losses

def eval_model_classification(model, graph, targetNode_mask, train_idx, valid_idx, test_idx):
    model = model.to(device)
    graph = graph.to(device)
    test_labels = to_cpu_npy(graph.y[targetNode_mask[test_idx]])
    
    model.eval()

    forward_scores = model(G)[targetNode_mask]

    test_scores = forward_scores[test_idx]
    test_softmax = F.softmax(test_scores, dim=1)
    test_preds = torch.argmax(test_softmax, dim=1)
    
    test_softmax = to_cpu_npy(test_softmax)
    test_preds = to_cpu_npy(test_preds)
    test_AUROC = roc_auc_score(test_labels, test_softmax[:,1], average="micro")
    test_acc = np.mean(test_preds == test_labels)

    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    train_scores = forward_scores[train_idx]
    train_softmax = F.softmax(train_scores, dim=1)
    train_preds = torch.argmax(train_softmax, dim=1)
    
    train_softmax = to_cpu_npy(train_softmax)
    train_preds = to_cpu_npy(train_preds)
    train_AUROC = roc_auc_score(train_labels, train_softmax[:,1], average="micro")
    train_acc = np.mean(train_preds == train_labels)


    return {'train_AUROC': train_AUROC, 'train_acc': train_acc, 'test_AUROC': test_AUROC, 'test_acc': test_acc}

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=3):
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
        self.dropout = nn.Dropout(p=0.5)

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

num_heads = 4
learning_rate = 0.002
max_epoch = 1500
loss = nn.CrossEntropyLoss()
hidden_channels=[6, 30]
wd = 1e-05

gat = GAT(in_channels=6, hidden_channels=hidden_channels, num_heads = num_heads)

optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, gat.parameters()), lr = learning_rate, weight_decay = wd)

train_losses, valid_losses = train_model_classification(gat, loss, G, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer)

out = eval_model_classification(gat, G, targetNode_mask, train_idx, valid_idx, test_idx)

print(out)







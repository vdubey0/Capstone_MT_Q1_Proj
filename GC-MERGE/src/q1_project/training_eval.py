# importing torch packages
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import GATConv
from torch_geometric.loader import NeighborLoader

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    forward_scores = model(graph)[targetNode_mask]

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

def calc_pearson(predictions, labels):
    return pearsonr(predictions, labels)[0]

def train_model_regression(model, graph, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer):
    '''
    Trains model for regression task
    
    Parameters
    ----------
    model [GCN_classification]: Instantiation of model class
    graph [PyG Data class]: PyTorch Geometric Data object representing the graph
    max_epoch [int]: Maximum number of training epochs
    learning_rate [float]: Learning rate
    targetNode_mask [tensor]: Subgraph mask for training nodes
    train_idx [array]: Node IDs corresponding to training set
    valid_idx [array]: Node IDs corresponding to validation set
    optimizer [PyTorch optimizer class]: PyTorch optimization algorithm

    Returns
    -------
    train_loss_vec [array]: Training loss for each epoch;
        analagous for valid_loss_vec (validation set)
    train_pearson_vec [array]: Training PCC for each epoch;
        analogous for valid_pearson_vec (validation set)
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    graph = graph.to(device)

    optimizer = optimizer
    
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    
    train_loss_list = []
    train_pearson_vec = np.zeros(np.shape(np.arange(max_epoch)))
    valid_loss_list = []
    valid_pearson_vec = np.zeros(np.shape(np.arange(max_epoch)))

    model.train()
    train_status = True
    
    print('\n')
    for e in list(range(max_epoch)):
        
        if e%100 == 0:
            print("Epoch", str(e), 'out of', str(max_epoch))
        
        model.train()
        train_status = True
        
        optimizer.zero_grad()
        
        ### Only trains on nodes with genes due to masking
        forward_scores = model(graph.x.float(), graph.edge_index, train_status)[targetNode_mask]
        
        train_scores = forward_scores[train_idx]

        train_loss  = model.loss(train_scores, torch.FloatTensor(train_labels).to(device))

        train_loss.backward()
        
        optimizer.step()
            
        ### Calculate training and validation loss, AUROC scores
        model.eval()
        
        train_scores = to_cpu_npy(train_scores)
        train_pearson = calc_pearson(train_scores, train_labels)
        train_loss_list.append(train_loss.item())
        
        valid_scores = forward_scores[valid_idx]
        valid_loss  = model.loss(valid_scores, torch.FloatTensor(valid_labels).to(device))
        valid_scores = to_cpu_npy(valid_scores)
        valid_pearson  = calc_pearson(valid_scores, valid_labels)
        valid_loss_list.append(valid_loss.item())
        
        train_pearson_vec[e] = train_pearson
        valid_pearson_vec[e] = valid_pearson

    train_loss_vec = np.reshape(np.array(train_loss_list), (-1, 1))
    valid_loss_vec = np.reshape(np.array(valid_loss_list), (-1, 1))

    return train_loss_vec, train_pearson_vec, valid_loss_vec, valid_pearson_vec


def eval_model_regression(model, graph, targetNode_mask, train_idx, valid_idx, test_idx):
    '''
    Runs fully trained regression model and compute evaluation statistics

    Parameters
    ----------
    model [GCN_classification]: Instantiation of model class
    graph [PyG Data class]: PyTorch Geometric Data object representing the graph
    targetNode_mask [tensor]: Mask ensuring model only trains on nodes with genes
    train_idx [array]: Node IDs corresponding to training set;
        analogous for valid_idx and test_idx

    Returns
    -------
    test_pearson [float]: PCC for test set;
        analogous for train_pearson (training set) and valid_pearson (validation set)
    test_pred [array]: Test set predictions;
        analogous for train_pred (training set) and valid_pred (validation set)
    test_labels [array]: Test set labels (expression values);
        analagous for train_labels (training set) and valid_labels (validation set)

    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    graph = graph.to(device)
    
    model.eval()
    train_status=False

    forward_scores = model(graph.x.float(), graph.edge_index, train_status)[targetNode_mask]

    test_scores = forward_scores[test_idx]
    test_pred = to_cpu_npy(test_scores)
    test_labels = to_cpu_npy(graph.y[targetNode_mask[test_idx]])
    test_pearson = calc_pearson(test_pred, test_labels)

    train_scores = forward_scores[train_idx]
    train_pred = to_cpu_npy(train_scores)
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    train_pearson = calc_pearson(train_pred, train_labels)

    valid_scores = forward_scores[valid_idx]
    valid_pred = to_cpu_npy(valid_scores)
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    valid_pearson = calc_pearson(valid_pred, valid_labels)

    return test_pearson, test_pred, test_labels, train_pearson, train_pred, train_labels, \
        valid_pearson, valid_pred, valid_labels
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
import torch_geometric.utils
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=5, dropout_rate=0.2):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels[0], heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels[0] * num_heads, hidden_channels[1], heads=num_heads, concat=True)
        self.gat3 = GATConv(hidden_channels[1] * num_heads, hidden_channels[2], heads=num_heads, concat=False)
        
        self.ff1 = nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2)
        self.ff2 = nn.Linear(hidden_channels[-1] // 2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.relu(self.gat3(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.relu(self.ff1(x))
        x = self.ff2(x)
        return x

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def to_cpu_npy(x):
    return x.cpu().detach().numpy()

def train_model_classification(model, loss, graph, max_epoch, targetNode_mask, train_idx, valid_idx, optimizer, device):
    model = model.to(device)
    graph = graph.to(device)

    optimizer = optimizer
    
    train_labels = graph.y[targetNode_mask[train_idx]].to(device)
    valid_labels = graph.y[targetNode_mask[valid_idx]].to(device)

    train_losses = []
    valid_losses = []

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()

        all_scores = model(graph)[targetNode_mask]
        train_scores = all_scores[train_idx]

        train_loss = loss(train_scores, train_labels)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            valid_scores = model(graph)[targetNode_mask]
            valid_loss = loss(valid_scores[valid_idx], valid_labels)
            valid_losses.append(valid_loss.item())

        if epoch % 100 == 0:  # Print every 100 epochs
            print(f'Epoch {epoch}: Train Loss = {train_loss}, Valid Loss = {valid_loss}')

    return train_losses, valid_losses

def eval_model_classification(model, graph, targetNode_mask, train_idx, test_idx, device):
    model = model.to(device)
    graph = graph.to(device)
    
    test_labels = graph.y[targetNode_mask[test_idx]].to(device)
    
    model.eval()

    forward_scores = model(graph)[targetNode_mask]

    test_scores = forward_scores[test_idx]
    test_softmax = F.softmax(test_scores, dim=1)
    test_preds = torch.argmax(test_softmax, dim=1)
    
    test_softmax = to_cpu_npy(test_softmax)
    test_preds = to_cpu_npy(test_preds)
    test_AUROC = roc_auc_score(to_cpu_npy(test_labels), test_softmax[:,1], average="micro")
    test_acc = np.mean(test_preds == test_labels)

    train_labels = graph.y[targetNode_mask[train_idx]].to(device)
    train_scores = forward_scores[train_idx]
    train_softmax = F.softmax(train_scores, dim=1)
    train_preds = torch.argmax(train_softmax, dim=1)
    
    train_softmax = to_cpu_npy(train_softmax)
    train_preds = to_cpu_npy(train_preds)
    train_AUROC = roc_auc_score(to_cpu_npy(train_labels), train_softmax[:,1], average="micro")
    train_acc = np.mean(train_preds == train_labels)


    return {'train_AUROC': train_AUROC, 'train_acc': train_acc, 'test_AUROC': test_AUROC, 'test_acc': test_acc}

def main(rank, world_size, dataset, targetNode_mask, train_idx, valid_idx, test_idx):
    ddp_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    else:
        print("No GPU available")

    inp_size = 6
    # hyperparameter tuning:
    hidden_sizes = [6, 24, 42]
    dropout_rate = 0.3
    n_heads = 3
    learning_rate = 0.01
    wd = 1e-05

    
    if rank == 0:
        print('\nGAT Model:')
        print('Input Size: ', inp_size)
        print('Hidden Size: ', hidden_sizes)
        print('Number of Heads: ', n_heads)
        print('Dropout Rate: ', dropout_rate)
        print('Learning Rate: ', learning_rate)
        print('\n')

    model = GAT(6, hidden_sizes, dropout_rate=dropout_rate, num_heads=n_heads).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)
    criterion = nn.CrossEntropyLoss()

    train_losses, valid_losses = train_model_classification(
        model,
        criterion,
        dataset,
        max_epoch=1500,
        targetNode_mask=targetNode_mask,
        train_idx=train_idx,
        valid_idx=valid_idx,
        optimizer=optimizer,
        device=device,
    )
    
    test_out = eval_model_classification(model, dataset, targetNode_mask, train_idx, test_idx, device)

    if rank == 0:
        print(f'\nTest AUROC: {test_out["test_AUROC"]}')

# Running the DDP training across multiple processes (2 GPUs)
if __name__ == '__main__':
    cell_line = 'E116'
    regression_flag = 0
    chip_res = 10000
    hic_res = 10000
    num_hm = 6
    num_feat = int((hic_res/chip_res)*num_hm)
    num_classes = 2 if regression_flag == 0 else 1

    # Define data paths
    src_dir = os.getcwd()
    save_dir = os.path.join(src_dir, 'data', cell_line, 'saved_runs')
    hic_sparse_mat_file = os.path.join(src_dir, 'data', cell_line, 'hic_sparse.npz')
    np_nodes_lab_genes_file = os.path.join(src_dir, 'data', cell_line, 'np_nodes_lab_genes_reg' + str(regression_flag) + '.npy')
    np_hmods_norm_all_file = os.path.join(src_dir, 'data', cell_line, 'np_hmods_norm_chip_' + str(chip_res) + 'bp.npy')
    df_genes_file = os.path.join(src_dir, 'data', cell_line, 'df_genes_reg' + str(regression_flag) + '.pkl')

    # Load data
    df_genes = pd.read_pickle(df_genes_file)
    mat = load_npz(hic_sparse_mat_file)
    allNodes_hms = np.load(np_hmods_norm_all_file)  # contains 6 histone marks for all 279606 regions + id (Shape = [279606, 7])
    hms = allNodes_hms[:, 1:]  # only includes features, not node ids (Shape = [279606, 6])
    X = torch.tensor(hms).float().reshape(-1, num_feat)  # convert hms to tensor (Shape = [279606, 6])
    allNodes = allNodes_hms[:, 0].astype(int)  # contains ids of all regions (Shape = [279606, 1])

    geneNodes_labs = np.load(np_nodes_lab_genes_file)  # contains the expression level of each gene (Shape = [16699, 2])
    geneNodes = geneNodes_labs[:, -2].astype(int)  # contains ids of regions that encode a gene (Shape = [16699, 1])
    allLabs = -1 * np.ones(np.shape(allNodes))
    geneLabs = geneNodes_labs[:, -1].astype(int)
    allLabs[geneNodes] = geneLabs

    targetNode_mask = torch.tensor(geneNodes).long()

    extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)
    data = torch_geometric.data.Data(edge_index=extract[0], edge_attr=extract[1], x=X, y=torch.tensor(allLabs).long())
    G = data

    train_idx = torch.load(f'train-test-split/{cell_line}/train_idx.pt')
    valid_idx = torch.load(f'train-test-split/{cell_line}/valid_idx.pt')
    test_idx = torch.load(f'train-test-split/{cell_line}/test_idx.pt')

    world_size = torch.cuda.device_count()  # Use all available GPUs
    mp.spawn(main, args=(world_size, G, targetNode_mask, train_idx, valid_idx, test_idx), nprocs=world_size, join=True)

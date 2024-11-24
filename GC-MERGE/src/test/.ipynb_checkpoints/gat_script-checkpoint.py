import os
import numpy as np
from scipy.sparse import load_npz
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import roc_auc_score

def setup_distributed():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = torch.cuda.device_count()
    torch.distributed.init_process_group(backend="nccl", world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    torch.distributed.destroy_process_group()

def to_cpu_npy(tensor):
    return tensor.cpu().detach().numpy()

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=3):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels[0], heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels[0] * num_heads, hidden_channels[1], heads=num_heads, concat=False)
        self.ff1 = nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2)
        self.ff2 = nn.Linear(hidden_channels[-1] // 2, 2)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.ff1(x))
        x = self.ff2(x)
        return x

def train_model(model, loss_fn, optimizer, loader, scaler, max_epoch, device):
    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with autocast():
                out = model(batch)
                loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{max_epoch}, Loss: {epoch_loss / len(loader):.4f}")

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast():
                out = model(batch)
                preds = torch.argmax(F.softmax(out[batch.test_mask], dim=1), dim=1)
                all_preds.append(to_cpu_npy(preds))
                all_labels.append(to_cpu_npy(batch.y[batch.test_mask]))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auroc = roc_auc_score(all_labels, all_preds, average="micro")
    acc = np.mean(all_preds == all_labels)
    return {"AUROC": auroc, "Accuracy": acc}

def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Load and preprocess data
    cell_line = 'E116'
    regression_flag = 0
    chip_res, hic_res, num_hm = 10000, 10000, 6
    num_feat = int((hic_res / chip_res) * num_hm)

    src_dir = os.getcwd()
    hic_sparse_mat_file = os.path.join(src_dir, 'data', cell_line, 'hic_sparse.npz')
    np_nodes_lab_genes_file = os.path.join(src_dir, 'data', cell_line, f'np_nodes_lab_genes_reg{regression_flag}.npy')
    np_hmods_norm_all_file = os.path.join(src_dir, 'data', cell_line, f'np_hmods_norm_chip_{chip_res}bp.npy')

    mat = load_npz(hic_sparse_mat_file)
    allNodes_hms = np.load(np_hmods_norm_all_file)
    hms = allNodes_hms[:, 1:]  # Features only
    X = torch.tensor(hms).float().reshape(-1, num_feat)
    allNodes = allNodes_hms[:, 0].astype(int)

    geneNodes_labs = np.load(np_nodes_lab_genes_file)
    geneNodes = geneNodes_labs[:, -2].astype(int)
    allLabs = -1 * np.ones(np.shape(allNodes))
    allLabs[geneNodes] = geneNodes_labs[:, -1].astype(int)
    Y = torch.tensor(allLabs).long()

    edge_index, edge_attr = torch_geometric.utils.from_scipy_sparse_matrix(mat)
    G = Data(edge_index=edge_index, edge_attr=edge_attr, x=X, y=Y)

    G.edge_index, _ = remove_self_loops(G.edge_index)
    num_edges = G.edge_index.size(1)
    perm = torch.randperm(num_edges)[:int(0.5 * num_edges)]
    G.edge_index = G.edge_index[:, perm]
    if G.edge_attr is not None:
        G.edge_attr = G.edge_attr[perm]

    pred_idx_shuff = torch.randperm(len(geneNodes))
    train_idx, valid_idx, test_idx = pred_idx_shuff[:int(0.7 * len(pred_idx_shuff))], pred_idx_shuff[int(0.7 * len(pred_idx_shuff)):int(0.85 * len(pred_idx_shuff))], pred_idx_shuff[int(0.85 * len(pred_idx_shuff)):]

    G.train_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    G.train_mask[train_idx] = True
    G.test_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    G.test_mask[test_idx] = True

    loader = NeighborLoader(G, num_neighbors=[10, 10], batch_size=1024, shuffle=True)

    gat = GAT(in_channels=num_feat, hidden_channels=[8, 16]).to(device)
    gat = DDP(gat, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(gat.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    train_model(gat, loss_fn, optimizer, loader, scaler, max_epoch=20, device=device)

    if local_rank == 0:
        results = evaluate_model(gat, loader, device)
        print(f"Test Results: {results}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
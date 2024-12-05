def train_model_classification_GAT(model, loss, graph, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer):
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

def eval_model_classification_GAT(model, graph, targetNode_mask, train_idx, valid_idx, test_idx):
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


def train_model_classification_GAT_Neighbors(model, loss, train_loader, valid_loader, max_epoch, optimizer, train_idx, valid_idx):
    model = model.to(device)

    train_losses = []
    valid_losses = []
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            train_batch_mask = torch.isin(batch.n_id.to(device), targetNode_mask.to(device))
            train_batch_scores = model(batch)[train_batch_mask]
            train_batch_labels = to_cpu_npy(batch.y[train_batch_mask])
            train_batch_loss = loss(train_batch_scores, torch.LongTensor(train_batch_labels).to(device))
            train_losses.append(train_batch_loss.item())
            train_batch_loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                valid_batch_mask = torch.isin(batch.n_id.to(device), targetNode_mask.to(device))
                valid_batch_scores = model(batch)[valid_batch_mask]
                valid_batch_labels = to_cpu_npy(batch.y[valid_batch_mask])
                valid_batch_loss = loss(valid_batch_scores, torch.LongTensor(valid_batch_labels).to(device))
                valid_losses.append(valid_batch_loss.item())
                
        print(f'Epoch {epoch}: Train Loss = {train_batch_loss}, Valid Loss = {valid_batch_loss}')
    return train_losses, valid_losses



def train_model_classification(model, graph, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer):
    '''
    Trains model for classification task
    
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
    train_loss_vec [array]: Training loss for each epoch
    train_AUROC_vec [array]: Training AUROC score for each epoch
    valid_loss_vec [array]: Validation loss for each epoch
    valid_AUROC_vec [array]: Validation AUROC score for each epoch

    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    graph = graph.to(device)

    optimizer = optimizer
    
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    
    train_loss_list = []
    train_AUROC_vec = np.zeros(np.shape(np.arange(max_epoch)))
    valid_loss_list = []
    valid_AUROC_vec = np.zeros(np.shape(np.arange(max_epoch)))

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

        train_loss  = model.loss(train_scores, torch.LongTensor(train_labels).to(device))

        train_softmax, _ = model.calc_softmax_pred(train_scores)

        train_loss.backward()
        
        optimizer.step()
            
        ### Calculate training and validation loss, AUROC scores
        model.eval()
        
        valid_scores = forward_scores[valid_idx]
        valid_loss  = model.loss(valid_scores, torch.LongTensor(valid_labels).to(device))
        valid_softmax, _ = model.calc_softmax_pred(valid_scores) 

        train_loss_list.append(train_loss.item())
        train_softmax = to_cpu_npy(train_softmax)
        train_AUROC = roc_auc_score(train_labels, train_softmax[:,1], average="micro")

        valid_loss_list.append(valid_loss.item())
        valid_softmax = to_cpu_npy(valid_softmax)
        valid_AUROC = roc_auc_score(valid_labels, valid_softmax[:,1], average="micro")
        
        train_AUROC_vec[e] = train_AUROC
        valid_AUROC_vec[e] = valid_AUROC

    train_loss_vec = np.reshape(np.array(train_loss_list), (-1, 1))
    valid_loss_vec = np.reshape(np.array(valid_loss_list), (-1, 1))

    return train_loss_vec, train_AUROC_vec, valid_loss_vec, valid_AUROC_vec


def eval_model_classification(model, graph, targetNode_mask, train_idx, valid_idx, test_idx):
    '''
    Runs fully trained classification model and compute evaluation statistics

    Parameters
    ----------
    model [GCN_classification]: Instantiation of model class
    graph [PyG Data class]: PyTorch Geometric Data object representing the graph
    targetNode_mask [tensor]: Mask ensuring model only trains on nodes with genes
    train_idx [array]: Node IDs corresponding to training set;
        analogous for valid_idx and test_idx

    Returns
    -------
    test_AUROC [float]: Test set AUROC score;
        analogous for train_AUROC (training set) and valid_AUPR (validation set)
    test_AUPR [float]: Test set AUPR score
        analogous for train_AUPR (training set) and valid_AUPR (validation set)
    test_pred [array]: Test set predictions;
        analogous for train_pred (training set) and valid_pred (validation set)
    test_labels [array]: Test set labels;
        analagous for train_labels (training set) and valid_labels (validation set)
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    graph = graph.to(device)
    test_labels = to_cpu_npy(graph.y[targetNode_mask[test_idx]])
    
    model.eval()
    train_status=False

    forward_scores = model(graph.x.float(), graph.edge_index, train_status)[targetNode_mask]

    test_scores = forward_scores[test_idx]
    test_softmax, test_pred = model.calc_softmax_pred(test_scores) 
    
    test_softmax = to_cpu_npy(test_softmax)
    test_pred = to_cpu_npy(test_pred)
    test_AUROC = roc_auc_score(test_labels, test_softmax[:,1], average="micro")
    test_precision, test_recall, thresholds = precision_recall_curve(test_labels, test_softmax[:,1])
    test_AUPR = auc(test_recall, test_precision)
    # test_F1 = f1_score(test_labels, test_pred, average="micro")
    
    train_scores = forward_scores[train_idx]
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    train_softmax, train_pred = model.calc_softmax_pred(train_scores) 
    train_pred = to_cpu_npy(train_pred)
    train_softmax = to_cpu_npy(train_softmax)
    train_precision, train_recall, thresholds = precision_recall_curve(train_labels, train_softmax[:,1])
    train_AUPR = auc(train_recall, train_precision)
    # train_F1 = f1_score(train_labels, train_pred, average="micro")

    valid_scores = forward_scores[valid_idx]
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    valid_softmax, valid_pred = model.calc_softmax_pred(valid_scores) 
    valid_pred = to_cpu_npy(valid_pred)
    valid_softmax = to_cpu_npy(valid_softmax)
    valid_precision, valid_recall, thresholds = precision_recall_curve(valid_labels, valid_softmax[:,1])
    valid_AUPR = auc(valid_recall, valid_precision)
    # valid_F1 = f1_score(valid_labels, valid_pred, average="micro")

    return test_AUROC, test_AUPR, test_pred, test_labels, train_AUPR, train_pred, train_labels, \
        valid_AUPR, valid_pred, valid_labels

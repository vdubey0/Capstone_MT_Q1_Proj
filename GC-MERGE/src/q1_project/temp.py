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
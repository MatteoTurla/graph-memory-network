import torch
from torch_geometric.utils import add_self_loops


def cluster_train(model, loader, criterion, optimizer, device):
    """
    train on N clustered subgraph of the same big graph. using clusterLoader utils
    """
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        train_mask = data.train_mask

        y = data.y.squeeze()[train_mask].to(device)

        out = model(x, edge_index)[train_mask]

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += train_mask.sum().item()
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example

# define graph batch train


@torch.no_grad()
def subgraph_test(model, subgraph_loader, data, device):
    """
    test on entire graph using inference method of the model
    """
    model.eval()

    y_true = data.y.squeeze()
    out = model.inference(data.x, subgraph_loader, device).cpu()

    train_mask = data.train_mask
    test_mask = data.test_mask

    train_acc = out[train_mask].argmax(dim=1).eq(
        y_true[train_mask]).sum().item() / train_mask.sum().item()
    test_acc = out[test_mask].argmax(dim=1).eq(
        y_true[test_mask]).sum().item() / test_mask.sum().item()

    return train_acc, test_acc


def batch_train(model, loader, criterion, optimizer, device):
    """
    train on N different small graph batched togheter using DataLoader utils
    we do not have a train mask but simply 2 different dataloader, one for train and one for test
    """
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        y = data.y.squeeze().to(device)

        out = model(x, edge_index)

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += y.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example


@torch.no_grad()
def batch_test(model, loader, device):
    model.eval()
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        y = data.y.squeeze().to(device)

        out = model(x, edge_index)

        total_example += y.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_correct/total_example

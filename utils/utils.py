import torch
from torch_geometric.utils import add_self_loops

def train(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = add_self_loops(data.edge_index)[0].to(device)
        
        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze()[data.train_mask].to(device)

        out = model(x, edge_index)[data.train_mask]

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += data.train_mask.sum().item()
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example

@torch.no_grad()
def test(model, subgraph_loader, data, device):
    model.eval()

    y_true = data.y.squeeze()
    out = model.inference(data.x, subgraph_loader, device)

    train_mask = data.train_mask
    test_mask = data.test_mask

    train_acc = out[train_mask].argmax(dim=1).eq(y_true[train_mask]).sum().item() / train_mask.sum().item()
    test_acc = out[test_mask].argmax(dim=1).eq(y_true[test_mask]).sum().item() / test_mask.sum().item()

    return train_acc, test_acc
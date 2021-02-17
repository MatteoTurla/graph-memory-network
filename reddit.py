from model.sageGMN import sageGMN
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import NeighborSampler
import torch


def train(model, laoder, criterion, optimizer, device):
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for batch in loader:
        batch_size, n_id, adjs = batch

        adjs = [adj.to(device) for adj in adjs]

        x = data.x[n_id].to(device)
        y = data.y[n_id[:batch_size]].squeeze().to(device)

        optimizer.zero_grad()

        out = model(x, adjs)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += batch_size
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example


@ torch.no_grad()
def test(model, subgraph_loader, data, device):
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


if __name__ == "__main__":

    dataset = Reddit(root='tmp/reddit', transform=NormalizeFeatures())

    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    data = dataset[0]
    train_idx = data.train_mask

    train_loader = NeighborSampler(data.edge_index, sizes=[25, 10, 10], node_idx=train_idx,
                                   batch_size=64, shuffle=True, return_e_id=False)

    # used for inference, look to github how to implement it
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=512, shuffle=False, return_e_id=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model = sageGMN(
        dataset.num_node_features, dataset.num_classes, n_heads=4, n_layers=3).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = train(
            model, train_loader, criterion, optimizer, device)
        if e % 3 == 0:
            train_acc, test_acc = test(
                model, subgraph_loader, data, device)
            print(loss, train_acc, test_acc)

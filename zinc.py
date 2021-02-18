from model.GMNregression import GMNregression
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import torch
from torch_geometric.transforms import NormalizeFeatures


def train(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze().to(device)

        out = model(x, edge_index, data.batch)

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1

    return total_loss/total_batch


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()

    total_batch = total_loss = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze().to(device)

        out = model(x, edge_index, data.batch)

        loss = criterion(out, y)

        total_loss += loss.item()
        total_batch += 1

    return total_loss/total_batch


if __name__ == "__main__":

    # graph regression

    dataset_train = ZINC(root='tmp/zinc', subset=True,
                         split="train",  pre_transform=NormalizeFeatures())
    dataset_val = ZINC(root='tmp/zinc', subset=True,
                       split="val",  pre_transform=NormalizeFeatures())

    print("number of features: ", dataset_train.num_features)
    print("number of classes: ", dataset_train.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # torch geomtric dataloader
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64)

    model = GMNregression(
        dataset_train.num_features, dataset_train.num_classes, 1, 5).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.MSELoss()

    for e in range(30):
        train_loss = train(
            model, train_loader, criterion, optimizer, device)
        if e % 10 == 0:
            val_loss = test(model, val_loader, criterion, device)
            print(train_loss, val_loss)

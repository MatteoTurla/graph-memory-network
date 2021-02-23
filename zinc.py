from model.GMN import GMNgraph
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops
import torch


def train(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = total_batches = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = add_self_loops(data.edge_index)[0].to(device)

        y = data.y.squeeze().to(device)

        batch = data.batch.to(device)

        out = model(x, edge_index, batch)

        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_batches += 1
        running_loss += loss.item()

    return running_loss / total_batches


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    running_loss = total_batches = 0.0
    y_hat = []
    y = []
    for data in loader:
        x = data.x.to(device)
        edge_index = add_self_loops(data.edge_index)[0].to(device)
        y = data.y.squeeze().to(device)
        batch = data.batch.to(device)
        out = model(x, edge_index, batch)

        loss = criterion(out, y)

        total_batches += 1
        running_loss += loss.item()

        y_hat.append(list(out.cpu()))
        y.append(list(y.cpu()))

    return running_loss / total_batches, y, y_hat


if __name__ == "__main__":

    dataset_train = ZINC(root='/tmp/', subset=True, split="train")
    dataset_val = ZINC(root='/tmp/', subset=True, split="val")

    n_features = dataset_train.num_features
    n_class = 1
    print("number of features: ", n_features)
    print("number of classes: ", n_class)
    print("n graphs train: ", len(dataset_train))
    print("n graphs val: ", len(dataset_val))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # torch geomtric dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=512)

    model = GMNgraph(
        n_features, n_class, 4, 3).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=0.0, lr=0.001)
    criterion = torch.nn.L1Loss()

    for e in range(30):
        loss = train(model, dataloader_train, optimizer, criterion, device)
        if e % 3 == 0:
            val_loss, _, _ = test(model, dataloader_val, device)
            print(loss, val_loss)

    val_loss, y, y_hat = test(model, dataloader_val, device)
    print(y, y_hat)

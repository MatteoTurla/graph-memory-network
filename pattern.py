from model.GraphMemoryNet import GraphMemoryNetwork
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import torch

# overwrite train function in utils, we do not have, mask there
def train(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = total_batch = 0.0
    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze().to(device)

        out = model(x, edge_index)

        optimizer.zero_grad()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1
        total_example += out.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_loss/total_batch, total_correct/total_example

@torch.no_grad()
def test(model, loader, device):
    model.eval()

    total_example = total_correct = 0.0

    for data in loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # in case y is a n x 1 tensor (ogb graph)
        y = data.y.squeeze().to(device)

        out = model(x, edge_index)

        total_example += out.shape[0]
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

    return total_correct/total_example

if __name__ == "__main__":

    dataset = GNNBenchmarkDataset(root='tmp/pattern', name="PATTERN")
    dataset_val = GNNBenchmarkDataset(root='tmp/pattern', name="PATTERN", split="val")

    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # torch geomtric dataloader
    dataloader = DataLoader(dataset, batch_size=64)
    dataloader_val = DataLoader(dataset_val, batch_size=64)
    
    model = GraphMemoryNetwork(dataset.num_features, 1, 5, dataset.num_classes).to(device)
    #print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = train(model, dataloader, criterion, optimizer, device)
        if e % 10 == 0:
            val_acc = test(model, dataloader_val, device)
            print(loss, train_acc, val_acc)
    
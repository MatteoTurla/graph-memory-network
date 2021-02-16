from model.GraphMemoryNet import GraphMemoryNetwork
from utils.utils import train, test
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import torch

if __name__ == "__main__":

    dataset = GNNBenchmarkDataset(root='tmp/pattern', name="PATTERN")

    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    dataloader = DataLoader(dataset, batch_size=32)
    
    model = GraphMemoryNetwork(dataset.num_features, 4, 2, dataset.num_classes).to(device)
    #print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = train(model, dataloader.train_loader, criterion, optimizer, device)
        print(loss, train_acc)
    
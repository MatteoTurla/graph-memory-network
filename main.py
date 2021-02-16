from model.GraphMemoryNet import GraphMemoryNetwork
from utils.DataLoader import DataLoader
from utils.utils import train, test
from torch_geometric.datasets import Planetoid
import torch

if __name__ == "__main__":

    name = "PubMed"
    dataset = Planetoid(root='./tmp/'+name, name=name)
    print("number of features: ", dataset.num_classes)

    dataloader = DataLoader(dataset, 0.4, 9, 3)
    
    model = GraphMemoryNetwork(128, 4, 2, 3)
    print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = train(model, dataloader.train_loader, criterion, optimizer, "cpu")
        if e % 3 == 0:
            train_acc, test_acc = test(model, dataloader.subgraph_loader, dataloader.graph, "cpu")
            print(loss, train_acc, test_acc)
    
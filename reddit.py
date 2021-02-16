from model.GraphMemoryNet import GraphMemoryNetwork
from utils.DataLoader import DataLoader
from utils.utils import train, test
from torch_geometric.datasets import Reddit
import torch

if __name__ == "__main__":

    dataset = Reddit(root='tmp/reddit')
    
    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    dataloader = DataLoader(128, 0.4, 100, 4, transform=True)
    
    model = GraphMemoryNetwork(128, 4, 3, dataset.num_classes).to(device)
    #print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = train(model, dataloader.train_loader, criterion, optimizer, device)
        if e % 3 == 0:
            train_acc, test_acc = test(model, dataloader.subgraph_loader, dataloader.graph, device)
            print(loss, train_acc, test_acc)
    
from model.GraphMemoryNet_2 import GraphMemoryNetwork, GraphMemoryNetwork_2
from utils.DataLoader import DataLoader
from utils.utils import cluster_train, test
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch

if __name__ == "__main__":

    name = "PubMed"
    dataset = Planetoid(root='tmp/'+name, name=name, transform=NormalizeFeatures())
    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # if test_size = None -> use standard mask
    dataloader = DataLoader(dataset, num_parts=128, batch_size=32, test_size=0.5)
    
    model = GraphMemoryNetwork_2(dataset.num_features, dataset.num_classes, n_heads=2, n_layers=3).to(device)
    #print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = cluster_train(model, dataloader.train_loader, criterion, optimizer, device)
        if e % 3 == 0:
            train_acc, test_acc = test(model, dataloader.subgraph_loader, dataloader.graph, device)
            print(loss, train_acc, test_acc)
    
from model.GraphMemoryNet_2 import GraphMemoryNetwork_2
from utils.DataLoader import DataLoader
from utils.utils import cluster_train, test
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import NormalizeFeatures
import torch

if __name__ == "__main__":

    dataset = Reddit(root='tmp/reddit', transform=NormalizeFeatures())

    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    dataloader = DataLoader(dataset, num_parts=1500, batch_size=32)

    model = GraphMemoryNetwork(
        dataset.num_edge_features, dataset.num_classes, n_heads=4, n_layers=3).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = cluster_train(
            model, dataloader.train_loader, criterion, optimizer, device)
        if e % 3 == 0:
            train_acc, test_acc = test(
                model, dataloader.subgraph_loader, dataloader.graph, device)
            print(loss, train_acc, test_acc)

from model.GMN import GMNnode
from utils.DataLoader import DataLoader
from utils.utils import cluster_train, subgraph_test
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch

if __name__ == "__main__":

    name = "Cora"
    dataset = Planetoid(root='tmp/'+name, name=name,
                        transform=NormalizeFeatures())
    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # if test_size = None -> use standard mask
    dataloader = DataLoader(dataset, num_parts=1,
                            batch_size=1, test_size=0.5)

    model = GMNnode(
        dataset.num_features, dataset.num_classes, n_heads=2, n_layers=3).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(300):
        loss, train_acc = cluster_train(
            model, dataloader.train_loader, criterion, optimizer, device)
        if e % 10 == 0:
            train_acc, test_acc = subgraph_test(
                model, dataloader.subgraph_loader, dataloader.graph, device)
            print(loss, train_acc, test_acc)

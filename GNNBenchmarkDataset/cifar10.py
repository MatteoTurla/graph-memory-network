from model.GMN import GMNgraph
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import torch
from utils import batch_train, batch_test

if __name__ == "__main__":

    name = "CIFAR10"
    dataset_train = GNNBenchmarkDataset(root='tmp/', name=name)
    dataset_val = GNNBenchmarkDataset(
        root='tmp/', name=name, split="val")

    print("number of features: ", dataset.num_features)
    print("number of classes: ", dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # torch geomtric dataloader
    dataloader_train = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=64)

    model = GMNgraph(
        dataset.num_features, dataset.num_classes, 1, 8).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = batch_train(
            model, dataloader_train, criterion, optimizer, device)
        if e % 10 == 0:
            val_acc = batch_test(model, dataloader_val, device)
            print(loss, train_acc, val_acc)

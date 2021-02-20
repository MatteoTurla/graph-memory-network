from model.GMN import GMNgraph
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import torch
from utils.utils import batch_train, batch_test

if __name__ == "__main__":

    name = "CIFAR10"
    dataset_train = GNNBenchmarkDataset(root='tmp/', name=name)
    dataset_val = GNNBenchmarkDataset(
        root='tmp/', name=name, split="val")

    n_features = dataset_train.num_features
    n_class = dataset_train.num_classes
    print("number of features: ", dataset_train.num_features)
    print("number of classes: ", dataset_train.num_classes)
    print("n graphs train: ", len(dataset_train))
    print("n graphs val: ", len(dataset_val))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # torch geomtric dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=256)

    model = GMNgraph(
        n_features, n_class, 4, 3).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(30):
        loss, train_acc = batch_train(
            model, dataloader_train, criterion, optimizer, device)
        if e % 10 == 0:
            val_acc = batch_test(model, dataloader_val, device)
            print(loss, train_acc, val_acc)

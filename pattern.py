from model.GMN import GMNnode
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import torch
from utils import batch_train, batch_test

if __name__ == "__main__":

    name = "pattern"
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
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=128)

    model = GraphMemoryNetwork(
        n_features, n_class, n_heads=4, n_layers=4).to(device)
    # print(model.train())

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(31):
        loss, train_acc = batch_train(
            model, dataloader_train, criterion, optimizer, device, reduce=False)
        if e % 10 == 0:
            val_acc = batch_test(model, dataloader_val, device, reduce=False)
            print(loss, train_acc, val_acc)

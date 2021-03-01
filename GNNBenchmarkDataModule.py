import pytorch_lightning as pl
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.transforms import AddSelfLoops,  Compose
from torch_geometric.data import DataLoader
from torch_geometric.utils import get_laplacian
import torch

class PositionalLaplacianEncoding(object):
    def __init__(self, k=2):
        self.k = k

    def __call__(self, data):
        num_nodes = data.y.shape[0]

        L = get_laplacian(
            data.edge_index, normalization="sym", num_nodes=num_nodes)

        L = torch.sparse.FloatTensor(
            L[0], L[1], size=(num_nodes, num_nodes)).to_dense()

        EigVal, EigVec = torch.eig(L, eigenvectors=True)
        idx = EigVal[:, 0].argsort()
        ordered_eigvec = EigVec[idx]
        pos_enc = ordered_eigvec[:, :self.k]

        data["pos_enc"] = pos_enc

        return data


class GNNBenchmarkDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name, batch_size=2, data_dir="/data/", k=2):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = Compose(
            [PositionalLaplacianEncoding(k), AddSelfLoops()])

    def prepare_data(self):

        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="train", pre_transform=self.transforms)
        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="val", pre_transform=self.transforms)
        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="test", pre_transform=self.transforms)

    def setup(self, stage):
        self.train_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                                 split="train", pre_transform=self.transforms)

        self.num_classes = self.train_dataset.num_classes
        self.num_features = self.train_dataset.num_features

        self.val_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                               split="val", pre_transform=self.transforms)

        self.test_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                                split="test", pre_transform=self.transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, batch_size=32, shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False)
        return test_loader

import pytorch_lightning as pl
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.data import DataLoader


class GNNBenchmarkDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name, batch_size=2, data_dir="/data/"):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="train", pre_transform=AddSelfLoops())
        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="val", pre_transform=AddSelfLoops())
        GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                            split="test", pre_transform=AddSelfLoops())

    def setup(self, stage):
        self.train_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                                 split="train", pre_transform=AddSelfLoops())

        self.num_classes = self.train_dataset.num_classes
        self.num_features = self.train_dataset.num_features

        self.val_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                               split="val", pre_transform=AddSelfLoops())

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, batch_size=256, shuffle=False)
        return val_loader

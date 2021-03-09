import pytorch_lightning as pl
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.transforms import AddSelfLoops,  Compose
from torch_geometric.data import DataLoader


"""
hyperparameters:
- number of eigenvector of laplacian positional encoding as described in gnnbenchamrking paper of xavier bresson
- add self loop 
"""

class GNNBenchmarkDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name, batch_size=2, data_dir="/data/", add_self_loops = False, positional_encoding=0):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        transformations = []
        if positional_encoding:
            transformations.append(PositionalLaplacianEncoding(positional_encoding))
        if add_self_loops:
            transformations.append(AddSelfLoops())
        if len(transformations):
            self.transforms = Compose(transformations)
        else:
            self.transforms = None

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

        self.val_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                               split="val", pre_transform=self.transforms)

        self.test_dataset = GNNBenchmarkDataset(root=self.data_dir, name=self.dataset_name,
                                                split="test", pre_transform=self.transforms)

        self.num_classes = self.train_dataset.num_classes
        self.num_features = self.train_dataset.num_features

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

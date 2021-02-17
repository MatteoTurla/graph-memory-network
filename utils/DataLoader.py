from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
import numpy as np
import torch

class DataLoader:
    def __init__(self, dataset, num_parts=1, batch_size=32, test_size=None,transform = False):
        
        self.dataset = dataset
        self.graph = dataset[0]
        self.n_nodes = self.graph.x.shape[0]

        if transform:
            self.graph.x = self.transform()

        # rewrite mask 
        if test_size is not None:
            idx_train, idx_test = train_test_split(
                np.array(list(range(self.n_nodes))),
                test_size = test_size,
                stratify = self.graph.y
            )

            self.graph.train_mask = self.create_mask(idx_train)
            self.graph.test_mask = self.create_mask(idx_test)

        self.cluster_data = ClusterData(self.graph, num_parts=num_parts, save_dir=self.dataset.processed_dir)
        self.train_loader = ClusterLoader(self.cluster_data, batch_size=batch_size, shuffle=True)

        # used for inference, look to github how to implement it
        self.subgraph_loader = NeighborSampler(self.graph.edge_index, sizes=[-1], 
                                      batch_size=512, shuffle=False, return_e_id=False)
            
    def create_mask(self, idx):
        mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        mask[idx] = 1

        return mask

    def transform(self):
        print("appling pca to planetoid dataset")
        x = self.graph.x.numpy()
        pca = PCA(n_components=128)
        x = pca.fit_transform(x)
        
        return torch.Tensor(x)
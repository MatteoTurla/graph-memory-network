import torch
from collections import OrderedDict
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from torch_geometric.nn import global_mean_pool


class GMNregression(torch.nn.Module):
    def __init__(self, input_size,  n_class, n_heads=1, n_layers=1):
        super().__init__()

        self.embedding = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(input_size, 128)),
            ('relu1', torch.nn.ReLU()),
            ('layer_norm', torch.nn.LayerNorm(128)),
        ]))

        self.layers = torch.nn.ModuleList(
            [MultiHeadAttentionLayer(128, n_heads) for i in range(n_layers)]
        )

        self.feed_forward = torch.nn.Sequential(OrderedDict([
            ('end_1', torch.nn.Linear(128, 512)),
            ('relu1', torch.nn.ReLU()),
            ('end_2', torch.nn.Linear(512, 512)),
            ('relu2', torch.nn.ReLU()),
            ('end_3', torch.nn.Linear(512, n_class))
        ]))

    def forward(self, X, edge_index, batch):

        X = self.embedding(X)

        for layer in self.layers:
            X = layer(X, edge_index)

        X = global_mean_pool(X, batch)

        y_hat = self.feed_forward(X)

        return y_hat

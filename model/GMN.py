import torch
from collections import OrderedDict
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.nn import global_mean_pool


class GMNnode(torch.nn.Module):
    """
    node property model
    """

    def __init__(self, input_size,  n_class, n_heads=1, n_layers=1):
        super().__init__()

        # as described in GNNbenchmarking papaer
        self.embedding = torch.nn.Linear(input_size, 128)

        self.layers = torch.nn.ModuleList(
            [MultiHeadAttentionLayer(128, n_heads) for i in range(n_layers)]
        )

        self.feed_forward = torch.nn.Sequential(OrderedDict([
            ('end_1', torch.nn.Linear(128, 128*2)),
            ('relu1', torch.nn.ReLU()),
            ('end_2', torch.nn.Linear(128, n_class))
        ]))

    def forward(self, X, edge_index):

        X = self.embedding(X)

        for layer in self.layers:
            X = layer(X, edge_index)

        y_hat = self.feed_forward(X)

        return y_hat

    def inference(self, x_all, subgraph_loader, device):
        x_all = x_all.to(device)
        x_all = self.embedding(x_all)

        # used to evaluate the model in a faster and memory efficient way
        for layer in self.layers:
            xs = []
            for batch_size, n_id, adj in subgraph_loader:

                edge_index, _, _ = adj
                # we are using neighbours sampler that return a direct bipartite graph
                edge_index = to_undirected(edge_index)
                edge_index = edge_index.to(device)

                x = x_all[n_id].to(device)
                x = layer(x, edge_index)[:batch_size]

                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        # now feed forward for the final classification
        y_hat = self.feed_forward(x_all)

        return y_hat


class GMNgraph(torch.nn.Module):
    """
    graph property model
    """

    def __init__(self, input_size,  n_class, n_heads=1, n_layers=1):
        super().__init__()

        self.embedding = torch.nn.Linear(input_size, 128)

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


class sageGMN(torch.nn.Module):
    """
    node property model
    aggregate using only a sample of neighbors
    """

    def __init__(self, input_size,  n_class, n_heads=1, n_layers=1):
        super().__init__()

        self.embedding = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(input_size, 128, bias=False)),
            # ('relu1', torch.nn.ReLU()),
            # ('layer_norm', torch.nn.LayerNorm(128)),
        ]))

        self.layers = torch.nn.ModuleList(
            [MultiHeadAttentionLayer(128, n_heads) for i in range(n_layers)]
        )

        self.feed_forward = torch.nn.Sequential(OrderedDict([
            ('end_1', torch.nn.Linear(128, 128*2)),
            ('relu1', torch.nn.ReLU()),
            ('end_2', torch.nn.Linear(128*2, n_class)),
            # ('relu2', torch.nn.ReLU()),
            # ('end_3', torch.nn.Linear(512, n_class))
        ]))

    def forward(self, X, adjs):

        X = self.embedding(X)

        for i, adj in enumerate(adjs):
            edge_index, _, size = adj
            X = self.layers[i](X, to_undirected(edge_index))[:size[1]]

        y_hat = self.feed_forward(X)

        return y_hat

    def inference(self, x_all, subgraph_loader, device):
        x_all = x_all.to(device)
        x_all = self.embedding(x_all)

        # used to evaluate the model in a faster and memory efficient way
        for layer in self.layers:
            xs = []
            for batch_size, n_id, adj in subgraph_loader:

                edge_index, _, _ = adj
                # we are using neighbours sampler that return a direct bipartite graph
                edge_index = to_undirected(edge_index)
                edge_index = edge_index.to(device)

                x = x_all[n_id].to(device)
                x = layer(x, edge_index)[:batch_size]

                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        # now feed forward for the final classification
        y_hat = self.feed_forward(x_all)

        return y_hat

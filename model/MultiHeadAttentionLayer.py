import torch
from collections import OrderedDict
from model.MemoryAggregator import MemoryAggregator


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, input_size, n_heads):
        super().__init__()

        self.input_size = input_size
        self.n_heads = n_heads

        self.heads = torch.nn.ModuleList(
            [MemoryAggregator(self.input_size, self.n_heads) for i in range(self.n_heads)])

        # multi-head matrix
        W0 = torch.Tensor(self.input_size, self.input_size)
        torch.nn.init.xavier_uniform_(W0)
        self.W0 = torch.nn.Parameter(W0)

        self.normalization_layer1 = torch.nn.LayerNorm(self.input_size)
        self.normalization_layer2 = torch.nn.LayerNorm(self.input_size)

        self.feed_forward = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(self.input_size, 512)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(512, self.input_size)),
        ]))

    def forward(self, X, edge_index):
        results = [head(X, edge_index) for head in self.heads]

        embedding = torch.matmul(torch.cat(results, dim=1), self.W0)
        add_normalize = self.normalization_layer1(X + embedding)

        ff = self.feed_forward(add_normalize)
        add_normalize2 = self.normalization_layer2(add_normalize + ff)

        return add_normalize2

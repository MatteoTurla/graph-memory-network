import torch
from collections import OrderedDict
from model.MemoryAggregator import MemoryAggregator


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, input_size, n_heads):
        super().__init__()

        self.heads = torch.nn.ModuleList(
            [MemoryAggregator(input_size, n_heads) for i in range(n_heads)])

        # multi-head matrix
        # do we need this?
        #self.W0 = torch.nn.Linear(input_size, input_size, bias=False)

        self.normalization_layer1 = torch.nn.LayerNorm(input_size)
        self.normalization_layer2 = torch.nn.LayerNorm(input_size)

        self.feed_forward = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(input_size, 128*2)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(128*2, input_size)),
        ]))

    def forward(self, X, edge_index):
        results = [head(X, edge_index) for head in self.heads]

        #embedding = self.W0(torch.cat(results, dim=1))
        embedding = torch.cat(results, dim=1)
        add_normalize = self.normalization_layer1(X + embedding)

        ff = self.feed_forward(add_normalize)
        add_normalize2 = self.normalization_layer2(add_normalize + ff)

        return add_normalize2

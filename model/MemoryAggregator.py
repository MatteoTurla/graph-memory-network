import torch
import math


class MemoryAggregator(torch.nn.Module):
    def __init__(self, input_size, n_heads):

        super().__init__()

        head_size = input_size // n_heads
        self.dk = math.sqrt(head_size)

        # query matrix
        self.Wq = torch.nn.Linear(input_size, head_size, bias=False)
        # key matrix
        self.Wk = torch.nn.Linear(input_size, head_size, bias=False)
        # value matrix
        self.Wv = torch.nn.Linear(input_size, head_size, bias=False)

    def forward(self, X, edge_index):
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        sparse_A = torch.sum(Q[edge_index[0, :]] *
                             K[edge_index[1, :]], axis=-1)

        dense_A = torch.sparse.FloatTensor(
            edge_index, sparse_A, torch.Size([X.shape[0], X.shape[0]]))

        masked_A = torch.div(dense_A, self.dk)

        attention = torch.sparse.mm(
            torch.sparse.softmax(masked_A, dim=1), V)

        return attention

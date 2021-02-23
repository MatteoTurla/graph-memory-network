import torch
import math


def stable_neighbors_softmax(a, edge_index, n_nodes):
    # a = Q_dot_Kt_stable
    a1 = a.exp()
    device = a.device
    a2 = torch.zeros(n_nodes, dtype=torch.float).to(device).scatter_add_(
        0, edge_index[0], a1)
    a3 = a2[edge_index[0]]
    return a1 / a3


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

        Q_dot_Kt = torch.sum(Q[edge_index[0, :]] *
                             K[edge_index[1, :]], axis=-1)
        # divide by the column dimension of K
        Q_dot_Kt_stable = torch.div(Q_dot_Kt, self.dk)

        masked_A = torch.sparse.FloatTensor(
            edge_index, Q_dot_Kt_stable, torch.Size([X.shape[0], X.shape[0]]))

        sparse_softmax_A = torch.sparse.softmax(masked_A, dim=1)
        # softmax_A = stable_neighbors_softmax(
        #    Q_dot_Kt_stable, edge_index, X.shape[0])
        # sparse_softmax_A = torch.sparse.FloatTensor(
        #   edge_index, softmax_A, torch.Size([X.shape[0], X.shape[0]]))

        attention = torch.sparse.mm(sparse_softmax_A, V)

        return attention

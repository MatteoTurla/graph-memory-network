import torch
import math

class MemoryAggregator(torch.nn.Module):
  def __init__(self, input_size, n_heads):

    super().__init__()

    self.input_size = input_size
    self.n_head = n_heads
    self.head_size = input_size // n_heads
    self.dk = math.sqrt(self.head_size)
    
    # query matrix
    Wq = torch.Tensor(self.input_size, self.head_size)
    torch.nn.init.xavier_uniform_(Wq)
    self.Wq = torch.nn.Parameter(Wq)
    # key matrix
    Wk= torch.Tensor(self.input_size, self.head_size)
    torch.nn.init.xavier_uniform_(Wk)
    self.Wk = torch.nn.Parameter(Wk)
    # value matrix
    Wv = torch.Tensor(self.input_size, self.head_size)
    torch.nn.init.xavier_uniform_(Wv)
    self.Wv = torch.nn.Parameter(Wv)

  def forward(self, X, edge_index):
    Q = torch.matmul(X, self.Wq)
    K = torch.matmul(X, self.Wk)
    V = torch.matmul(X, self.Wv)

    r = torch.cat((edge_index[0,:], edge_index[1,:]))
    c = torch.cat((edge_index[1,:], edge_index[0,:]))

    sparse_A = torch.sum(Q[r] * K[c], axis=-1)
    dense_A = torch.sparse.FloatTensor(torch.vstack((r,c)), sparse_A)

    masked_A = torch.div(dense_A, self.dk)

    attention = torch.matmul(torch.sparse.softmax(masked_A, dim=1).to_dense(), V)

    return attention

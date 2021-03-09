import torch
from torch import nn
from torch.nn import functional as F
import math

class GNNconfig:
    # drop out config
    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    num_layers = None
    num_heads = None

    num_classes = None
    input_dim = None
    embedding_dim = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class AttentionAggregator(nn.Module):

    def __init__(self, config):
        super().__init__()

        embedding_dim = config.embedding_dim
        num_heads = config.num_heads

        assert embedding_dim % num_heads == 0

        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # affine transformation from 2F -> F, where F = embedding dimension
        # proj( x_i^(l-1) || neighbour_aggregation )
        self.proj = nn.Linear(embedding_dim * 2, embedding_dim)

        # parameter used in forward method
        self.nunm_heads = num_heads
        self.dim_head = embedding_dim // num_heads
        self.dk = math.sqrt(self.dim_head)

    def forward(self, x, edge_index):

        num_nodes, embed_dim = x.size()

        k = self.key(x).view(num_nodes, self.nunm_heads,
                             self.dim_head).transpose(0, 1)
        q = self.query(x).view(num_nodes, self.nunm_heads,
                               self.dim_head).transpose(0, 1)
        v = self.value(x).view(num_nodes, self.nunm_heads,
                               self.dim_head).transpose(0, 1)

        att = (q @ k.transpose(-2, -1)) * (1.0 / self.dk)

        # create the neighours mask
        mask = torch.ones((x.shape[0], x.shape[0]),
                          dtype=torch.bool, device=edge_index.device, requires_grad=False)
        mask[edge_index[0, :], edge_index[1, :]] = False

        att = att.masked_fill(mask == True, float('-inf'))
        att = self.attn_drop(att)
        att = F.softmax(att, dim=-1)

        y = att @ v

        y = y.transpose(0, 1).contiguous().view(num_nodes, embed_dim)

        y = self.proj(torch.cat((x, y), dim=1))
        # it is possible yo add a dropout layer there

        return y


class Block(nn.Module):
    """
    basic layer block:
    h_l = h_l-1 + activation(normalization(aggregator(normalize(x), edge_index)))
    residual connection and normalization should stabilize gradient and converge faster
    """
    def __init__(self, config):
        super().__init__()

        embedding_dim = config.embedding_dim
        norm = config.norm

        # take a look to the last papaer about normalization in graph neural network and add them!
        if norm == "layer":
            self.ln1 = nn.LayerNorm(embedding_dim)
            self.ln2 = nn.LayerNorm(embedding_dim)
        elif norm == "batch":
            self.ln1 = nn.BatchNorm1d(embedding_dim)
            self.ln2 = nn.BatchNorm1d(embedding_dim)
        else:
            raise Exception("norm must be layer or batch")

        self.aggregator = NeighborsAttention(config)
        self.activation = nn.ReLU()

    def forward(self, data):
        x = data.x
        residual = x
        edge_index = data.edge_index

        x = self.aggregator(self.ln1(x), edge_index)
        x = residual + self.activation(self.ln2(x))
        # it is possible to add a dropout layer there

        data.x = x
        return data


class GNN(nn.Module):
    """
    boilerplate of a gnn 
    x -> embedding -> aggregator_blocks -> task dependent layer
    """
    def __init__(self, config):
        super().__init__()

        input_dim = config.input_dim
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers
        num_classes = config.num_classes

        self.embedding = nn.Linear(input_dim, embedding_dim)

        # aggregator blocks
        self.blocks = nn.Sequential(*[Block(config)
                                      for _ in range(num_layers)])

        self.mlp = nn.Sequential(
            # gtp-3 use a normalization layer also there
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, num_classes),
        )
 

        # init weights as gtp
        if config.init_weights == "custom":
            self.apply(self._init_weights)

    # init weights as GTP-3
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, data):

        data.x = self.embedding(data.x)
        # it is possible to add a dropout layer there

        data = self.blocks(data)

        logits = self.mlp(data.x)

        return logits

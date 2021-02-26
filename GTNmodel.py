import torch
from torch import nn
from torch.nn import functional as F
import math

"""
TO DO: weight init as in gpt-3, config adam optimizer and lr decay, positional encoding of nodes
"""


class GTNconfig:
    """ base config """
    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class NeighborsAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        embedding_dim = config.embedding_dim
        num_heads = config.num_heads
        attn_pdrop = config.attn_pdrop
        resid_pdrop = config.resid_pdrop

        assert embedding_dim % num_heads == 0

        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(embedding_dim, embedding_dim)

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

        mask = torch.ones((x.shape[0], x.shape[0]),
                          dtype=torch.bool, device=edge_index.device, requires_grad=False)
        mask[edge_index[0, :], edge_index[1, :]] = False

        att = att.masked_fill(mask == True, float('-inf'))
        att = self.attn_drop(att)
        att = F.softmax(att, dim=-1)

        y = att @ v

        y = y.transpose(0, 1).contiguous().view(num_nodes, embed_dim)

        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        embedding_dim = config.embedding_dim
        resid_pdrop = config.resid_pdrop

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.attn = NeighborsAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = x + self.attn(self.ln1(x), edge_index)
        x = x + self.mlp(self.ln2(x))

        data.x = x
        return data


class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config.input_dim
        embedding_dim = config.embedding_dim
        num_layers = config.num_layers
        num_classes = config.num_classes

        self.embedding = nn.Linear(input_dim, embedding_dim)
        # we should add a graph embedding and a dropout

        # transformer layer
        self.blocks = nn.Sequential(*[Block(config)
                                      for _ in range(num_layers)])

        # feed forward layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, num_classes),
        )

    def forward(self, data):

        data.x = self.embedding(data.x)
        data = self.blocks(data)

        logits = self.mlp(data.x)

        return logits

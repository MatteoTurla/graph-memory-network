import torch
from torch import nn
from torch.nn import functional as F


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

    def forward(self, data):

        x = data.x
        # can or can not include self loops
        edge_index = data.edge_index

        num_nodes, embed_dim = x.size()

        k = self.key(x).view(num_nodes, self.nunm_heads,
                             embed_dim // self.nunm_heads).transpose(0, 1)
        q = self.query(x).view(num_nodes, self.nunm_heads,
                               embed_dim // self.nunm_heads).transpose(0, 1)
        v = self.value(x).view(num_nodes, self.nunm_heads,
                               embed_dim // self.nunm_heads).transpose(0, 1)

        att = (q1 @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        mask = torch.ones((x.shape[0], x.shape[0]), dtype=torch.bool)
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

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


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
        x = data.x
        x = self.embedding(x)
        x = self.blocks(data)

        logits = self.mlp(x)

        return logits

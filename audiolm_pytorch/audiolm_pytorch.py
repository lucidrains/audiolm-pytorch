import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from vector_quantize_pytorch import VectorQuantize as VQ

# helper functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# sound stream

class SoundStream(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 10
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d'), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# audio LM

class AudioLM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

import math

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from vector_quantize_pytorch import VectorQuantize as VQ

# helper functions

def exists(val):
    return val is not None

# sound stream

class SoundStream(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0

        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, i, j, device):

        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)

        rel_pos = k_pos[None, :] - q_pos[:, None]

        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)

        return rearrange(values, 'i j h -> h i j')

# feedforward

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias()

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, **kwargs),
                FeedForward(dim = dim)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        n, device = x.shape[1], x.device

        rel_pos_bias = self.rel_pos_bias(n, n, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = rel_pos_bias) + x
            x = ff(x) + x

        return self.norm(x)

# audio LM

class AudioLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        **kwargs
    ):
        super().__init__()
        self.attend_semantic = Transformer(dim = dim, depth = depth, **kwargs)
        self.attend_coarse = Transformer(dim = dim, depth = depth, **kwargs)
        self.attend_fine = Transformer(dim = dim, depth = depth, **kwargs)

    def forward(self, x):
        x = self.attend_semantic(x)
        return x

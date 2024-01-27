import math
from functools import partial, wraps

from beartype.typing import Optional, Union, List
from beartype import beartype

import torch
from torch import nn, einsum, Tensor
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torchaudio

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans

from audiolm_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from torchaudio.functional import resample

from audiolm_pytorch.soundstream import SoundStream
from audiolm_pytorch.encodec import EncodecWrapper
from audiolm_pytorch.utils import AudioConditionerBase
from audiolm_pytorch.attend import Attend

from tqdm import tqdm
from pathlib import Path
from audiolm_pytorch.version import __version__
from packaging import version

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def maybe(fn):
    if not exists(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

def remainder_needed_until_multiple(n, mult):
    return (ceil_div(n, mult) * mult) - n

def round_down_nearest_multiple(val, mult):
    return (val // mult) * mult

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

# attention related utils

def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def mask_out_after_eos_id(t, eos_id, mask_value = -1, keep_eos = True):
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim = -1) > 0
    return t.masked_fill(after_eos_mask, mask_value)

def all_rows_have_eos_id(t, eos_id):
    eos_mask = (t == eos_id)
    return torch.any(eos_mask, dim = -1).all()

def safe_cat(*tensors, dim = -2):
    args = [*filter(exists, tensors)]

    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return torch.cat(args, dim = dim)

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# removing unique consecutives in the semantic token ids
# important detail noted by @eonglints

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device = device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b = b)
    ids = torch.cat((ids, eos_ids), dim = -1)
    return ids

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

# function for getting embeds from nn.Embedding but with padding as some designated value (-1) outside the range of the embed table

@beartype
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # just retrieve first code as dummy
    embeds = embeddings(codes_without_pad)

    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    if return_mask:
        return embeds, ~pad_mask

    return embeds

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# relative positional bias

class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert j >= i
        device = self.device

        i_pos = torch.arange(i, device = device) + (j - i)
        j_pos = torch.arange(j, device = device)

        rel_pos = (rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j'))
        rel_pos += (j - 1)

        x = torch.arange(-j + 1, j, device = device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        num_null_kv = 0,
        dropout = 0.1,
        scale = 8,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        prefix_context = None,
        prefix_context_mask = None,
        return_kv_cache = False,
        kv_cache = None
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim = -2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device = device, dtype = torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim = -1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value = 0.)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # kv cache

        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # store kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # split for multi-headed attention

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # handle mask and null key / value

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)

        # attention

        out = self.attend(q, k, v, attn_bias = attn_bias, mask = mask)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, kv_cache

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context = None,
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        grad_shrink_alpha = 0.1,
        cond_as_self_attn_prefix = False,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        assert not (cross_attend and cond_as_self_attn_prefix)

        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha = grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(dim = dim // 2, heads = heads) if rel_pos_bias else None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dropout = attn_dropout, flash = flash_attn, causal = True, **kwargs),
                Attention(dim = dim, heads = heads, dropout = attn_dropout, dim_context = dim_context, flash = flash_attn, num_null_kv = 1, norm_context = True, **kwargs) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        self_attn_mask = None,
        context = None,
        context_mask = None,
        attn_bias = None,
        return_kv_cache = False,
        kv_cache = None
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        x = self.grad_shrink(x)

        # turn off kv cache if using conditioning as self attention (as in valle), for now

        if self.cond_as_self_attn_prefix:
            kv_cache = None

        # handle kv cache

        new_kv_cache = []

        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])

        x = x[:, cache_len:]

        # relative positional bias

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)

        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]

        # self attention kwargs

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context = context,
                prefix_context_mask = context_mask
            )

        # transformer layers

        for attn, cross_attn, ff in self.layers:

            residual = x

            x, layer_kv_cache = attn(x, attn_bias = rel_pos_bias, mask = self_attn_mask, kv_cache = next(kv_cache, None), return_kv_cache = True, **self_attn_kwargs)
            new_kv_cache.append(layer_kv_cache)

            x = x + residual

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context = context, mask = context_mask) + x

            x = ff(x) + x

        x = self.norm(x)

        if not return_kv_cache:
            return x

        return x, torch.stack(new_kv_cache)

# the three hierarchical transformers

class SemanticTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        cond_dim = None,
        has_condition = False,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        self.num_semantic_tokens = num_semantic_tokens

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.start_token = nn.Parameter(torch.randn(dim))

        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        self.eos_id = num_semantic_tokens

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )

        self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        kv_cache = None,
        return_kv_cache = False,
        **kwargs
    ):
        kv_cache = iter(default(kv_cache, []))
        new_kv_caches = []

        logits, new_kv_cache = self.forward(*args, cond_drop_prob = 0., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(new_kv_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return logits

            return logits, torch.stack(new_kv_caches)

        null_logits, null_new_kv_cache = self.forward(*args, cond_drop_prob = 1., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(null_new_kv_cache)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if not return_kv_cache:
            return scaled_logits

        return scaled_logits, torch.stack(new_kv_caches)

    @beartype
    def forward(
        self,
        *,
        ids = None,
        return_loss = False,
        text: Optional[List[str]] = None,
        text_embeds = None,
        self_attn_mask = None,
        cond_drop_prob = None,
        unique_consecutive = None,
        kv_cache = None,
        return_kv_cache = False
    ):
        device = self.device

        b = ids.shape[0]

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        if return_loss:
            labels, ids = ids.clone(), ids[:, :-1]

        tokens = get_embeds(self.semantic_embedding, ids)

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = ids.shape[0])

        tokens = torch.cat((start_tokens, tokens), dim = 1)

        if exists(self_attn_mask):
            self_attn_mask = F.pad(self_attn_mask, (1, 0), value = True)

        tokens, kv_cache = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask, kv_cache = kv_cache, return_kv_cache = True)
        logits = self.to_logits(tokens)

        if not return_kv_cache:
            return logits

        return logits, kv_cache

class CoarseTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_coarse_quantizers,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_semantic_logits = True,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        self.num_semantic_tokens = num_semantic_tokens

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.semantic_start_token = nn.Parameter(torch.randn(dim))
        self.coarse_start_token = nn.Parameter(torch.randn(dim))

        self.semantic_eos_id = num_semantic_tokens
        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)

        self.coarse_eos_id = codebook_size
        codebook_size_with_eos = codebook_size + 1

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size_with_eos, dim)
        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.cross_attn_bias = nn.Parameter(torch.zeros(heads, 1, 1)) if rel_pos_bias else None

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers

        self.to_semantic_logits = nn.Linear(dim, num_semantic_tokens + 1) if project_semantic_logits else None
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
    ):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []

        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    @beartype
    def forward(
        self,
        *,
        semantic_token_ids,
        coarse_token_ids,
        self_attn_mask = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        return_only_coarse_logits = False,
        return_cache = False,
        kv_cache = None,
        embed_cache = None
    ):
        b, device = semantic_token_ids.shape[0], semantic_token_ids.device
        arange = partial(torch.arange, device = device)

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)

        text_mask = None
        if exists(text_embeds):
            text_mask = torch.any(text_embeds != 0, dim = -1)

            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids, semantic_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, semantic_token_ids))

        offsets = self.codebook_size * arange(self.num_coarse_quantizers)
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        offsets = offsets[:, :coarse_token_ids.shape[-1]]
        coarse_token_ids = coarse_token_ids + offsets

        semantic_tokens = get_embeds(self.semantic_embedding, semantic_token_ids)
        coarse_tokens = self.coarse_embedding(coarse_token_ids)

        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens

        semantic_seq_len = semantic_tokens.shape[1]

        semantic_start_tokens = repeat(self.semantic_start_token, 'd -> b 1 d', b = b)
        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            semantic_start_tokens,
            semantic_tokens,
            coarse_start_tokens,
            coarse_tokens
        ), dim = 1)

        # engineer the attention bias so that cross attention is not dominated by relative positions

        seq_len = tokens.shape[-2]

        attn_bias = None

        if exists(self.transformer.rel_pos_bias):
            attn_bias = self.transformer.rel_pos_bias(seq_len, seq_len)

            is_semantic = arange(seq_len) < (semantic_seq_len + 1) # semantic seq len + start token
            is_cross_attn = rearrange(is_semantic, 'i -> i 1') ^ rearrange(is_semantic, 'j -> 1 j')

            attn_bias = torch.where(
                is_cross_attn,
                self.cross_attn_bias,
                attn_bias
            )

        # attend

        tokens, new_kv_cache = self.transformer(
            tokens,
            context = text_embeds,
            attn_bias = attn_bias,
            self_attn_mask = self_attn_mask,
            context_mask = text_mask,
            kv_cache = kv_cache,
            return_kv_cache = True
        )

        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim = -2)

        new_embed_cache = tokens

        # segment into semantic and coarse acoustic tokens

        pred_semantic_tokens, pred_coarse_tokens = tokens[:, :semantic_seq_len], tokens[:, (semantic_seq_len + 1):]

        # semantic logits

        semantic_logits = self.to_semantic_logits(pred_semantic_tokens) if not return_only_coarse_logits and exists(self.to_semantic_logits) else None

        # get coarse logits

        n = pred_coarse_tokens.shape[1]
        nq = round_down_nearest_multiple(n, self.num_coarse_quantizers)

        pred_coarse_tokens_groupable, pred_coarse_tokens_remainder = pred_coarse_tokens[:, :nq], pred_coarse_tokens[:, nq:]

        pred_coarse_tokens_groupable = rearrange(pred_coarse_tokens_groupable, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits_groupable = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens_groupable)

        coarse_logits_groupable = rearrange(coarse_logits_groupable, 'b n q c -> b (n q) c')

        remainder_num_quantizers = pred_coarse_tokens_remainder.shape[1]

        if remainder_num_quantizers > 0:
            coarse_logits_remainder = einsum('q c d, b q d -> b q c', self.coarse_logit_weights[:remainder_num_quantizers], pred_coarse_tokens_remainder)

            coarse_logits = torch.cat((coarse_logits_groupable, coarse_logits_remainder), dim = 1)
        else:
            coarse_logits = coarse_logits_groupable

        logits = (semantic_logits, coarse_logits)

        if not return_cache:
            return logits

        return logits, (new_kv_cache, new_embed_cache)

class FineTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_coarse_quantizers,
        num_fine_quantizers,
        codebook_size,
        dim,
        depth,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_coarse_logits = True,
        pad_id = -1,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.num_coarse_quantizers = num_coarse_quantizers

        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        self.fine_start_token = nn.Parameter(torch.randn(dim))

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size, dim)
        self.fine_embedding = nn.Embedding(num_fine_quantizers * codebook_size, dim)

        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)
        self.fine_quantize_embedding = nn.Embedding(num_fine_quantizers, dim)

        self.pad_id = pad_id
        self.eos_id = codebook_size

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            rel_pos_bias = False,
            grad_shrink_alpha = grad_shrink_alpha,
            flash_attn = flash_attn,
            **kwargs
        )

        # doing a specialized attn bias so that corresponding time steps at fine and coarse sequences attend to each other better

        self.null_pos_bias = nn.Parameter(torch.randn(heads, 1, 1)) if rel_pos_bias else None

        pos_bias_mlp_dim = dim // 2

        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(2, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, pos_bias_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_bias_mlp_dim, heads)
        ) if rel_pos_bias else None

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers

        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size, dim)) if project_coarse_logits else None
        self.fine_logit_weights = nn.Parameter(torch.randn(num_fine_quantizers, codebook_size, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
    ):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []

        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    def forward(
        self,
        coarse_token_ids,
        fine_token_ids,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        self_attn_mask = None,
        kv_cache = None,
        embed_cache = None,
        return_cache = False,
        return_only_fine_logits = False
    ):
        b, device = coarse_token_ids.shape[0], coarse_token_ids.device

        # handle text conditioning

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids, fine_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, fine_token_ids))

        # do not attend to any of the coarse padding tokens or coarse end token either

        coarse_self_attn_mask = (coarse_token_ids != self.pad_id) & (coarse_token_ids != self.eos_id)
        coarse_token_ids = coarse_token_ids.masked_fill(~coarse_self_attn_mask, 0)

        fine_token_seq_len = fine_token_ids.shape[-1]
        coarse_self_attn_mask = F.pad(coarse_self_attn_mask, (1, fine_token_seq_len + 1), value = True)

        if exists(self_attn_mask):
            self_attn_mask &= coarse_self_attn_mask
        else:
            self_attn_mask = coarse_self_attn_mask

        # prepare coarse and fine token embeddings

        b, n = coarse_token_ids.shape

        coarse_length = coarse_token_ids.shape[-1]
        coarse_offsets = torch.arange(self.num_coarse_quantizers, device = device)
        coarse_seq_length = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers)
        coarse_offsets = repeat(coarse_offsets, 'q -> (n q)', n = coarse_seq_length)
        coarse_offsets = coarse_offsets[:coarse_length]
        coarse_token_ids = coarse_token_ids + rearrange(coarse_offsets, '... -> 1 ...') * self.codebook_size

        fine_length = fine_token_ids.shape[-1]
        fine_offsets = torch.arange(self.num_fine_quantizers, device = device)
        fine_seq_length = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers)
        fine_offsets = repeat(fine_offsets, 'q -> (n q)', n = fine_seq_length)
        fine_offsets = fine_offsets[:fine_length]
        fine_token_ids = fine_token_ids + rearrange(fine_offsets, '... -> 1 ...') * self.codebook_size

        coarse_tokens = self.coarse_embedding(coarse_token_ids)
        fine_tokens = self.fine_embedding(fine_token_ids)

        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens

        fine_quantize_tokens = repeat(self.fine_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers))
        fine_quantize_tokens = fine_quantize_tokens[:fine_token_ids.shape[-1], ...]
        fine_tokens = fine_tokens + fine_quantize_tokens

        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)
        fine_start_tokens = repeat(self.fine_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            coarse_start_tokens,
            coarse_tokens,
            fine_start_tokens,
            fine_tokens
        ), dim = 1)

        # an engineered attention bias so coarse and fine sequences attend to each other better

        attn_bias = None

        if exists(self.pos_bias_mlp):
            max_seq_len = max(coarse_seq_length, fine_seq_length)

            coarse_pos = torch.arange(coarse_seq_length, device = device)
            fine_pos = torch.arange(fine_seq_length, device = device)

            coarse_pos = repeat(coarse_pos, 'n -> (n q)', q = self.num_coarse_quantizers)[:coarse_length]
            fine_pos = repeat(fine_pos, 'n -> (n q)', q = self.num_fine_quantizers)[:fine_length]

            coarse_pos = F.pad(coarse_pos, (1, 0), value = -1)
            fine_pos = F.pad(fine_pos, (1, 0), value = -1)

            seq_positions = torch.cat((coarse_pos, fine_pos), dim = -1)

            coarse_offsets = F.pad(coarse_offsets, (1, 0), value = 0)
            fine_offsets = fine_offsets + self.num_coarse_quantizers
            fine_offsets = F.pad(fine_offsets, (1, 0), value = 0)

            seq_offsets = torch.cat((coarse_offsets, fine_offsets), dim = -1)

            pos_mlp_input = torch.stack((seq_positions.clamp(min = 0), seq_offsets), dim = -1)

            num_offsets = self.num_fine_quantizers + self.num_coarse_quantizers

            # relative positions are always (2 * N - 1), where N is the length of the dimension

            rel_seq_len, rel_offsets = map(lambda n: 2 * n - 1, (max_seq_len, num_offsets))

            # get all relative distances

            rel_dist = (rearrange(pos_mlp_input, 'i c -> i 1 c') - rearrange(pos_mlp_input, 'j c -> 1 j c'))

            # get all possible relative distances for the attention bias to be computed from the mlp
            # which would be - (2 * N - 1) * (2 * Q - 1) - where N = sequence length and Q = total quantizers

            rel_seq_len_range = repeat(torch.arange(rel_seq_len, device = device), 'n -> (n q)', q = rel_offsets)
            rel_offset_range = repeat(torch.arange(rel_offsets, device = device), 'q -> (n q)', n = rel_seq_len)

            mlp_inputs = torch.stack((rel_seq_len_range, rel_offset_range), dim = -1)

            # implicitly parameterized relative distances, by sequence and quantizer positions

            attn_bias = self.pos_bias_mlp(mlp_inputs.float())

            # translate coordinates of (rel_seq_pos, rel_quantizer_offset) -> positive index to select from attn bias

            rel_dist_seq_pos, rel_dist_seq_offset = rel_dist.unbind(dim = -1)

            rel_dist_seq_pos += max_seq_len - 1
            rel_dist_seq_offset += num_offsets - 1

            rel_dist_indices = rel_dist_seq_pos * rel_offsets + rel_dist_seq_offset

            # select the relative positional attention bias outputted by the MLP
            # savings go from (N * Q) ^ 2 -> ~ (4 * N * Q)

            attn_bias = attn_bias[rel_dist_indices]

            attn_bias = rearrange(attn_bias, '... h -> h ...')

            # need to make sure start token has a custom positional bias

            is_start_token_seq = seq_positions == -1
            start_token_mask = rearrange(is_start_token_seq, 'i -> i 1') | rearrange(is_start_token_seq, 'j -> 1 j')

            attn_bias = torch.where(
                start_token_mask,
                self.null_pos_bias,
                attn_bias,
            )

        # attention

        tokens, next_kv_cache = self.transformer(
            tokens,
            context = text_embeds,
            self_attn_mask = self_attn_mask,
            context_mask = text_mask,
            attn_bias = attn_bias,
            kv_cache = kv_cache,
            return_kv_cache = True
        )

        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim = -2)

        new_embed_cache = tokens

        # figure out which tokens are coarse vs fine for logit projection

        pred_coarse_tokens, pred_fine_tokens = tokens[:, :n], tokens[:, (n + 1):]

        # get coarse logits

        pred_coarse_seq_len = pred_coarse_tokens.shape[1]

        padding = remainder_needed_until_multiple(pred_coarse_seq_len, self.num_coarse_quantizers)

        if padding != 0:
            pred_coarse_tokens = F.pad(pred_coarse_tokens, (0, 0, 0, padding), value = 0.)

        pred_coarse_tokens = rearrange(pred_coarse_tokens, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits = None

        if not return_only_fine_logits and exists(self.coarse_logit_weights):
            coarse_logits = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens)

            coarse_logits = rearrange(coarse_logits, 'b n q c -> b (n q) c')

            coarse_logits = coarse_logits[:, :pred_coarse_seq_len]

        # get fine logits

        pred_fine_seq_len = pred_fine_tokens.shape[1]
        nq = round_down_nearest_multiple(pred_fine_seq_len, self.num_fine_quantizers)

        pred_fine_tokens_groupable, pred_fine_tokens_remainder = pred_fine_tokens[:, :nq], pred_fine_tokens[:, nq:]

        pred_fine_tokens_groupable = rearrange(pred_fine_tokens_groupable, 'b (n q) d -> b n q d', q = self.num_fine_quantizers)

        fine_logits_groupable = einsum('q c d, b n q d -> b n q c', self.fine_logit_weights, pred_fine_tokens_groupable)

        fine_logits_groupable = rearrange(fine_logits_groupable, 'b n q c -> b (n q) c')

        remainder_num_quantizers = pred_fine_tokens_remainder.shape[1]

        if remainder_num_quantizers > 0:
            fine_logits_remainder = einsum('q c d, b q d -> b q c', self.fine_logit_weights[:remainder_num_quantizers], pred_fine_tokens_remainder)

            fine_logits = torch.cat((fine_logits_groupable, fine_logits_remainder), dim = 1)
        else:
            fine_logits = fine_logits_groupable

        logits = (coarse_logits, fine_logits)

        if not return_cache:
            return logits

        return logits, (next_kv_cache, new_embed_cache)

# training wrappers

class SemanticTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        pad_id = -1,
        unique_consecutive = True,
        mask_prob = 0.15
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        assert not exists(self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_text(self, text):
        return self.transformer.embed_text(text, output_device = self.device)

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        max_length,
        text: Optional[List[str]] = None,
        text_embeds = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_ids = None,
        batch_size = 1,
        cond_scale = 3,
        filter_thres = 0.9,
        temperature = 1.,
        use_kv_cache = True,
        include_eos_in_output = True,  # if doing hierarchical sampling, eos must be kept for an easy time
        **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            ids = self.wav2vec(
                prime_wave,
                flatten = False,
                input_sample_hz = prime_wave_input_sample_hz
            )
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # derive joint audio-text embeddings if needed

        if exists(self.audio_conditioner) and exists(prime_wave):
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = prime_wave, namespace = 'semantic')

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()

        # kv cache

        kv_cache = None
        logits = None

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            new_logits, new_kv_cache = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                cond_scale = cond_scale,
                kv_cache = kv_cache,
                return_kv_cache = True,
                **kwargs
            )

            if use_kv_cache:
                kv_cache = new_kv_cache
                logits = safe_cat(logits, new_logits, dim = -2)
            else:
                logits = new_logits

            last_logit_indices_expanded = repeat(last_logit_indices, 'b -> b 1 c', b = batch, c = logits.shape[-1])
            last_logits = logits.gather(1, last_logit_indices_expanded)

            last_logits = rearrange(last_logits, 'b 1 c -> b c')

            filtered_logits = top_k(last_logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            sample_semantic_ids = torch.cat((sample_semantic_ids, sampled), dim = -1)

            if all_rows_have_eos_id(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1

        sample_semantic_ids = mask_out_after_eos_id(sample_semantic_ids, self.eos_id, keep_eos = False)

        return sample_semantic_ids

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        text = None,
        text_embeds = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'semantic')

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

        input_ids = semantic_token_ids
        if return_loss:
            input_ids = semantic_token_ids[:, :-1]

        self_attn_mask = None
        if self.mask_prob > 0. and self.training:
            self_attn_mask = generate_mask_with_prob(input_ids.shape, self.mask_prob, input_ids.device)

        logits = self.transformer(
            ids = input_ids,
            text = text,
            text_embeds = text_embeds,
            self_attn_mask = self_attn_mask,
            **kwargs
        )

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            semantic_token_ids,
            ignore_index = self.pad_id
        )

        return loss

class CoarseTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        codec: Optional[Union[SoundStream, EncodecWrapper]]  = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        pad_id = -1,
        unique_consecutive = True,
        semantic_cross_entropy_loss_weight = 1.,
        mask_prob = 0.15
    ):
        super().__init__()
        self.codec = codec
        self.wav2vec = wav2vec

        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight

        self.num_coarse_quantizers = transformer.num_coarse_quantizers * codec.rq_groups
        self.semantic_eos_id = transformer.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids,
        prime_wave: Optional[Tensor] = None,
        prime_wave_input_sample_hz = None,
        prime_coarse_token_ids: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        max_time_steps = 512,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        use_kv_cache = True,
        **kwargs
    ):
        batch, device = semantic_token_ids.shape[0], self.device

        semantic_token_ids = semantic_token_ids.to(device)

        # initialize coarse token ids
        # if a prime audio wave was supplied, then start off with appropriate acoustic tokens

        assert not (exists(prime_wave) and exists(prime_coarse_token_ids)), 'you can either pass in the prime as a raw wave (codec required) or as preprocessed acoustic token ids'

        if exists(prime_coarse_token_ids):
            coarse_token_ids = prime_coarse_token_ids
        elif exists(prime_wave):
            assert exists(self.codec)
            with torch.inference_mode():
                self.codec.eval()

                _, indices, _ = self.codec(
                    prime_wave,
                    return_encoded = True,
                    input_sample_hz = prime_wave_input_sample_hz
                )

                coarse_token_ids = indices[..., :self.num_coarse_quantizers]
                coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        else:
            coarse_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value=self.pad_id)

        # initialize

        init_coarse_time_step = 0
        sampled_coarse_token_ids = coarse_token_ids.clone()

        # kv cache

        kv_cache = None
        embed_cache = None

        for time_step in tqdm(range(init_coarse_time_step, max_time_steps), desc = 'generating coarse'):
            for ind in range(self.num_coarse_quantizers):
                just_finished_quantizer_step = (ind == 0 and time_step > 0)

                (_, coarse_logits), (next_kv_cache, next_embed_cache) = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = sampled_coarse_token_ids,
                    semantic_token_ids = semantic_token_ids,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_kv_cache = True,
                    kv_cache = kv_cache,
                    embed_cache = embed_cache,
                    return_only_coarse_logits = True,
                    **kwargs
                )

                if use_kv_cache:
                    kv_cache = next_kv_cache
                    embed_cache = next_embed_cache

                last_coarse_logits = coarse_logits[:, -1]

                if not just_finished_quantizer_step:
                    last_coarse_logits[:, -1] = float('-inf') # prevent from eos in the middle of a time step

                filtered_logits = top_k(last_coarse_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat((sampled_coarse_token_ids, sampled), dim = -1)

        sampled_coarse_token_ids = mask_out_after_eos_id(sampled_coarse_token_ids, self.coarse_eos_id, keep_eos = False)
        sampled_coarse_token_ids = rearrange(sampled_coarse_token_ids, 'b (n q) -> b n q', q = self.num_coarse_quantizers)

        if not reconstruct_wave:
            return sampled_coarse_token_ids

        assert exists(self.codec)

        coarse_tokens_are_variable_lengthed = (sampled_coarse_token_ids == -1).any()

        if not coarse_tokens_are_variable_lengthed:
            wav = self.codec.decode_from_codebook_indices(sampled_coarse_token_ids)
            return rearrange(wav, 'b 1 n -> b n')

        # handle variable lengthed coarse tokens

        wavs = []
        for coarse_sample in sampled_coarse_token_ids:
            has_padding = reduce(coarse_sample == -1, 'n q -> n', 'any')
            coarse_sample_without_padding = coarse_sample[~has_padding]

            if has_padding.all():
                wavs.append(None)
                continue

            coarse_sample_without_padding = rearrange(coarse_sample_without_padding, '... -> 1 ...')

            wav = self.codec.decode_from_codebook_indices(coarse_sample_without_padding)
            wav = rearrange(wav, '1 1 n -> n')

            wavs.append(wav)

        return wavs

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        raw_wave_for_codec = None,
        text = None,
        text_embeds = None,
        coarse_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        raw_wave_for_codec = default(raw_wave_for_codec, raw_wave)
        assert exists(raw_wave_for_codec) or exists(coarse_token_ids), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        assert not all(map(exists, (raw_wave, raw_wave_for_codec, semantic_token_ids, coarse_token_ids)))

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'coarse') # technically audio embeds, but shared text-audio joint embedding space for mulan

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        if not exists(coarse_token_ids):
            assert exists(self.codec), 'Codec must be provided if given raw wave for training'

            with torch.inference_mode():
                self.codec.eval()
                _, indices, _ = self.codec(raw_wave_for_codec, return_encoded = True)

                batch, num_timesteps = raw_wave_for_codec.shape
                num_frames = int(num_timesteps / self.codec.seq_len_multiple_of)

                assert indices.shape[0] == batch and indices.shape[1] == num_frames, \
                    f'Expected indices to have shape (batch, num_frames, num_coarse_quantizers + num_fine_quantizers), but got {indices.shape}'

                coarse_token_ids = indices[..., :self.num_coarse_quantizers]

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.semantic_eos_id)
            coarse_token_ids = append_eos_id(coarse_token_ids, self.transformer.coarse_eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

        if return_loss:
            semantic_labels, coarse_labels = semantic_token_ids, coarse_token_ids.clone()
            coarse_token_ids = coarse_token_ids[:, :-1]

        # self attention mask would omit any padding and eos tokens in the semantic prime

        self_attn_mask = (semantic_token_ids != self.pad_id) & (semantic_token_ids != self.semantic_eos_id)
        semantic_token_ids = semantic_token_ids.masked_fill(~self_attn_mask, 0)

        coarse_token_len = coarse_token_ids.shape[-1]
        self_attn_mask = F.pad(self_attn_mask, (1, coarse_token_len + 1), value = True) # attend to semantic bos and all coarse tokens

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(self_attn_mask.shape, self.mask_prob, device = self_attn_mask.device)

        semantic_logits, coarse_logits = self.transformer(
            semantic_token_ids = semantic_token_ids,
            coarse_token_ids = coarse_token_ids,
            self_attn_mask = self_attn_mask,
            text = text,
            text_embeds = text_embeds,
            **kwargs
        )

        # whether to early return the logits

        if not return_loss:
            return semantic_logits, coarse_logits

        coarse_logits, semantic_logits = map(lambda t: maybe(rearrange)(t, 'b n c -> b c n'), (coarse_logits, semantic_logits))

        if self.unique_consecutive:
            num_coarse_logits, _num_semantic_logits = coarse_labels.numel(), (semantic_labels != self.pad_id).sum()
        else:
            num_coarse_logits, _num_semantic_logits = coarse_logits.shape[-1], semantic_logits.shape[-1]

        semantic_loss = 0.
        num_semantic_logits = 0

        if self.semantic_cross_entropy_loss_weight > 0 and exists(semantic_logits):
            num_semantic_logits = _num_semantic_logits

            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index = self.pad_id
            )

        coarse_loss = F.cross_entropy(
            coarse_logits,
            coarse_labels,
            ignore_index = self.pad_id
        )

        return (
            semantic_loss * num_semantic_logits * self.semantic_cross_entropy_loss_weight +
            coarse_loss * num_coarse_logits
        ) / (num_semantic_logits + num_coarse_logits)

class FineTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: FineTransformer,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        coarse_cross_entropy_loss_weight = 1.,
        pad_id = -1,
        mask_prob = 0.15
    ):
        super().__init__()
        self.codec = codec

        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        self.num_fine_quantizers = transformer.num_fine_quantizers * codec.rq_groups
        self.num_coarse_quantizers = transformer.num_coarse_quantizers * codec.rq_groups

        if exists(codec):
            assert (self.num_fine_quantizers + self.num_coarse_quantizers) == (codec.num_quantizers * codec.rq_groups), 'number of fine and coarse quantizers on fine transformer must add up to total number of quantizers on codec'

        self.eos_id = transformer.eos_id

        assert self.num_coarse_quantizers > 0

        self.pad_id = pad_id
        self.coarse_cross_entropy_loss_weight = coarse_cross_entropy_loss_weight

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        coarse_token_ids,
        prime_wave: Optional[Tensor] = None,
        prime_wave_input_sample_hz = None,
        prime_fine_token_ids: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        use_kv_cache = True,
        mask_out_generated_fine_tokens = False,
        **kwargs
    ):
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        batch, device = coarse_token_ids.shape[0], self.device

        coarse_token_ids = coarse_token_ids.to(device)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # initialize fine token ids
        # if a prime wave was supplied, start off with fine acoustic tokens

        assert not (exists(prime_wave) and exists(prime_fine_token_ids)), 'you can either pass in the prime as a raw wave (codec required) or as preprocessed acoustic token ids'

        if exists(prime_fine_token_ids):
            fine_token_ids = prime_fine_token_ids
        elif exists(prime_wave):
            assert exists(self.codec)
            with torch.inference_mode():
                self.codec.eval()
                _, token_ids, _ = self.codec(
                    prime_wave,
                    return_encoded = True,
                    input_sample_hz = prime_wave_input_sample_hz
                )

            fine_token_ids = token_ids[..., self.num_coarse_quantizers:]
            fine_token_ids = rearrange(fine_token_ids, 'b ... -> b (...)')
        else:
            fine_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        # calculate number of sampling steps

        init_fine_time_step = fine_token_ids.shape[-1] // self.num_fine_quantizers
        max_time_steps = coarse_token_ids.shape[1] // self.num_coarse_quantizers

        sampled_fine_token_ids = fine_token_ids.clone()

        # kv cache

        kv_cache = None
        embed_cache = None

        for time_step in tqdm(range(init_fine_time_step, max_time_steps), desc = 'generating fine'):
            for ind in range(self.num_fine_quantizers):
                just_finished_quantizer_step = (ind == 0 and time_step > 0)

                (_, fine_logits), (next_kv_cache, next_embed_cache) = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = coarse_token_ids,
                    fine_token_ids = sampled_fine_token_ids,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_only_fine_logits = True,
                    kv_cache = kv_cache,
                    embed_cache = embed_cache,
                    return_kv_cache = True,
                    **kwargs
                )

                last_fine_logits = fine_logits[:, -1]

                if use_kv_cache:
                    kv_cache = next_kv_cache
                    embed_cache = next_embed_cache

                if not just_finished_quantizer_step:
                    last_fine_logits[:, -1] = float('-inf')  # prevent from eos in the middle of a time step

                filtered_logits = top_k(last_fine_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_fine_token_ids = torch.cat((sampled_fine_token_ids, sampled), dim = -1)

        sampled_fine_token_ids = mask_out_after_eos_id(sampled_fine_token_ids, self.eos_id, keep_eos = False)

        # reshape coarse and fine tokens for quantization dimension

        sampled_fine_token_ids = rearrange(sampled_fine_token_ids, 'b (n q) -> b n q', q = self.num_fine_quantizers)
        coarse_token_ids = rearrange(coarse_token_ids, 'b (n q) -> b n q', q = self.num_coarse_quantizers)

        # whether to mask out fine token positions where the coarse token ids are all padding (variable lengthed training)

        if mask_out_generated_fine_tokens:
            pos_is_all_padding = (coarse_token_ids == self.pad_id).all(dim = -1, keepdim = True)
            sampled_fine_token_ids = sampled_fine_token_ids.masked_fill(pos_is_all_padding, self.pad_id)

        # if not reconstructing wave, return just the fine token ids

        if not reconstruct_wave:
            return sampled_fine_token_ids

        # reconstruct the wave using codec, concatting the fine and coarse token ids together first across quantization dimension

        assert exists(self.codec)

        coarse_and_fine_ids = torch.cat((coarse_token_ids, sampled_fine_token_ids), dim = -1)

        # need to handle padding (uneven acoustic token lengths)

        has_any_pad_mask = reduce(coarse_and_fine_ids == self.pad_id, 'b n q -> b n', 'any')

        if not has_any_pad_mask.any():
            wav = self.codec.decode_from_codebook_indices(coarse_and_fine_ids)
            return rearrange(wav, 'b 1 n -> b n')

        # naively decode each sample at a time if padding exists

        wavs = []

        for acoustic_ids_with_padding, pad_mask in zip(coarse_and_fine_ids, has_any_pad_mask):
            acoustic_ids = acoustic_ids_with_padding[~pad_mask]
            acoustic_ids = rearrange(acoustic_ids, 'n q -> 1 n q')
            wav = self.codec.decode_from_codebook_indices(acoustic_ids)
            wav = rearrange(wav, '1 1 n -> n')
            wavs.append(wav)

        return wavs

    def forward(
        self,
        *,
        raw_wave = None,
        text = None,
        text_embeds = None,
        token_ids = None,
        coarse_token_ids = None,
        fine_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) ^ (exists(token_ids) ^ (exists(coarse_token_ids) and exists(fine_token_ids))), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'fine') # technically audio embeds, but shared text-audio joint embedding space for mulan

        if exists(raw_wave):
            assert exists(self.codec), 'Codec must be provided if given raw wave for training'

            with torch.inference_mode():
                self.codec.eval()
                _, token_ids, _ = self.codec(raw_wave, return_encoded = True)

                batch, num_timesteps = raw_wave.shape
                num_frames = int(num_timesteps / self.codec.seq_len_multiple_of)

                assert token_ids.shape == torch.Size((batch, num_frames, self.num_coarse_quantizers + self.num_fine_quantizers)), \
                    f'Expected token ids to have shape (batch, num_frames, num_coarse_quantizers + num_fine_quantizers), but got {token_ids.shape}'

        if exists(token_ids):
            coarse_token_ids, fine_token_ids = token_ids[..., :self.num_coarse_quantizers], token_ids[..., self.num_coarse_quantizers:]

        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        fine_token_ids = rearrange(fine_token_ids, 'b ... -> b (...)')

        # if training, determine labels, should remove one from fine token ids

        if return_loss:
            coarse_labels = coarse_token_ids
            fine_labels = fine_token_ids
            fine_token_ids = fine_token_ids[:, :-1]

        # forgetful causal mask - structured dropout

        self_attn_mask = None

        if self.mask_prob > 0 and self.training:
            mask_shape = (
                coarse_token_ids.shape[0],
                coarse_token_ids.shape[-1] + fine_token_ids.shape[-1] + 2
            )

            self_attn_mask = generate_mask_with_prob(mask_shape, self.mask_prob, device = self.device)

        coarse_logits, fine_logits = self.transformer(
            coarse_token_ids = coarse_token_ids,
            fine_token_ids = fine_token_ids,
            self_attn_mask = self_attn_mask,
            text = text,
            text_embeds = text_embeds,
            **kwargs
        )

        # early return the logits

        if not return_loss:
            return coarse_logits, fine_logits

        coarse_logits, fine_logits = map(lambda t: maybe(rearrange)(t, 'b n c -> b c n'), (coarse_logits, fine_logits))

        num_fine_logits = fine_logits.shape[-1]

        num_coarse_logits = 0
        coarse_loss = 0.

        if self.coarse_cross_entropy_loss_weight > 0 and exists(coarse_logits):
            num_coarse_logits = coarse_logits.shape[-1]

            coarse_loss = F.cross_entropy(
                coarse_logits,
                coarse_labels,
                ignore_index = self.pad_id
            )

        fine_loss = F.cross_entropy(
            fine_logits,
            fine_labels,
            ignore_index = self.pad_id
        )

        return (
            coarse_loss * num_coarse_logits * self.coarse_cross_entropy_loss_weight +
            fine_loss * num_fine_logits
        ) / (num_coarse_logits + num_fine_logits)

# audio LM

class AudioLM(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]], 
        codec: Union[SoundStream, EncodecWrapper],
        semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        audio_conditioner: Optional[AudioConditionerBase] = None,
        unique_consecutive = True
    ):
        super().__init__()

        self.audio_conditioner = audio_conditioner

        assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers
        assert (fine_transformer.num_coarse_quantizers + fine_transformer.num_fine_quantizers) == codec.num_quantizers

        self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])

        self.semantic = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = semantic_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        self.coarse = CoarseTransformerWrapper(
            wav2vec = wav2vec,
            codec = codec,
            transformer = coarse_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        self.fine = FineTransformerWrapper(
            codec= codec,
            transformer = fine_transformer,
            audio_conditioner = audio_conditioner
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    def forward(
        self,
        *,
        batch_size = 1,
        text: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_wave_path = None,
        max_length = 2048,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False
    ):
        assert not (self.needs_text and (not exists(text) and not exists(text_embeds))), 'text needs to be passed in if one of the transformer requires conditioning'

        if self.needs_text:
            if exists(text):
                text_embeds = self.semantic.embed_text(text)

        assert not (exists(prime_wave) and exists(prime_wave_path)), 'prompt audio must be given as either `prime_wave: Tensor` or `prime_wave_path: str`'

        if exists(prime_wave):
            assert exists(prime_wave_input_sample_hz), 'the input sample frequency for the prompt audio must be given as `prime_wave_input_sample_hz: int`'
            prime_wave = prime_wave.to(self.device)
        elif exists(prime_wave_path):
            prime_wave_path = Path(prime_wave_path)
            assert exists(prime_wave_path), f'file does not exist at {str(prime_wave_path)}'

            prime_wave, prime_wave_input_sample_hz = torchaudio.load(str(prime_wave_path))
            prime_wave = prime_wave.to(self.device)

        semantic_token_ids = self.semantic.generate(
            text_embeds = text_embeds if self.semantic_has_condition else None,
            batch_size = batch_size,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            max_length = max_length
        )

        coarse_token_ids_or_recon_wave = self.coarse.generate(
            text_embeds = text_embeds if self.coarse_has_condition else None,
            semantic_token_ids = semantic_token_ids,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = return_coarse_generated_wave
        )

        if return_coarse_generated_wave:
            return coarse_token_ids_or_recon_wave

        generated_wave = self.fine.generate(
            text_embeds = text_embeds if self.fine_has_condition else None,
            coarse_token_ids = coarse_token_ids_or_recon_wave,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = True,
            mask_out_generated_fine_tokens = mask_out_generated_fine_tokens
        )

        return generated_wave

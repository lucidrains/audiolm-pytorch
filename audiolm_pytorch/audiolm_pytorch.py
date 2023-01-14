import math
from functools import partial

from beartype.typing import Optional, Union, List
from beartype import beartype

import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans

from audiolm_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from torchaudio.functional import resample

from audiolm_pytorch.soundstream import SoundStream

from tqdm import tqdm

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

class CausalDSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv1d(dim, dim, 3, bias = False, groups = dim)

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = F.pad(x, (2, 0))
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        CausalDSConv(inner_dim * 2),
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
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
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
        prefix_context_mask = None
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

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # split for multi-headed attention

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        # similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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
        **kwargs
    ):
        super().__init__()
        assert not (cross_attend and cond_as_self_attn_prefix)
        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha = grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(heads = heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dropout = attn_dropout, causal = True, **kwargs),
                Attention(dim = dim, heads = heads, dropout = attn_dropout, dim_context = dim_context, num_null_kv = 1, norm_context = True, **kwargs) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        self_attn_mask = None,
        context = None,
        context_mask = None
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))

        n, device = x.shape[1], x.device

        x = self.grad_shrink(x) # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        rel_pos_bias = self.rel_pos_bias(n, n, device = device)

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context = context,
                prefix_context_mask = context_mask
            )

        for attn, cross_attn, ff in self.layers:
            x = attn(x, attn_bias = rel_pos_bias, mask = self_attn_mask, **self_attn_kwargs) + x

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context = context, mask = context_mask)

            x = ff(x) + x

        return self.norm(x)

# the three hierarchical transformers

@beartype
class SemanticTransformer(nn.Module):
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
        has_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        pad_id = -1,
        **kwargs
    ):
        super().__init__()
        self.num_semantic_tokens = num_semantic_tokens

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.start_token = nn.Parameter(torch.randn(dim))

        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        self.eos_id = num_semantic_tokens
        self.pad_id = pad_id

        text_dim = get_encoded_dim(t5_name)
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
            **kwargs
        )

        self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        *,
        ids = None,
        return_loss = False,
        text: Optional[List[str]] = None,
        text_embeds = None,
        self_attn_mask = None,
        cond_drop_prob = None,
        unique_consecutive = None
    ):
        device = self.device

        b = ids.shape[0]

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
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

        tokens = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask)
        return self.to_logits(tokens)

@beartype
class CoarseTransformer(nn.Module):
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
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_semantic_tokens = num_semantic_tokens

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

        text_dim = get_encoded_dim(t5_name)
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
            **kwargs
        )

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers

        self.to_semantic_logits = nn.Linear(dim, num_semantic_tokens + 1)
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        semantic_logits, coarse_logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return semantic_logits, coarse_logits

        null_semantic_logits, null_coarse_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale
        return scaled_semantic_logits, scaled_coarse_logits

    def forward(
        self,
        *,
        semantic_token_ids,
        coarse_token_ids,
        self_attn_mask = None,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        return_only_coarse_logits = False
    ):
        b, device = semantic_token_ids.shape[0], semantic_token_ids.device

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
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

        offsets = self.codebook_size * torch.arange(self.num_coarse_quantizers, device = device)
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        offsets = offsets[:, :coarse_token_ids.shape[-1]]
        coarse_token_ids = coarse_token_ids + offsets

        semantic_tokens = self.semantic_embedding(semantic_token_ids)
        coarse_tokens = self.coarse_embedding(coarse_token_ids)

        semantic_seq_len = semantic_tokens.shape[1]

        semantic_start_tokens = repeat(self.semantic_start_token, 'd -> b 1 d', b = b)
        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            semantic_start_tokens,
            semantic_tokens,
            coarse_start_tokens,
            coarse_tokens
        ), dim = 1)

        tokens = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask)

        pred_semantic_tokens, pred_coarse_tokens = tokens[:, :semantic_seq_len], tokens[:, (semantic_seq_len + 1):]

        # semantic logits

        semantic_logits = self.to_semantic_logits(pred_semantic_tokens) if not return_only_coarse_logits else None

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

        return semantic_logits, coarse_logits

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
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        **kwargs
    ):
        super().__init__()
        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.num_coarse_quantizers = num_coarse_quantizers

        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        self.fine_start_token = nn.Parameter(torch.randn(dim))

        codebook_size_with_eos = codebook_size + 1

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size_with_eos, dim)
        self.fine_embedding = nn.Embedding(num_fine_quantizers * codebook_size_with_eos, dim)

        self.eos_id = codebook_size

        text_dim = get_encoded_dim(t5_name)
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
            **kwargs
        )

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers

        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))
        self.fine_logit_weights = nn.Parameter(torch.randn(num_fine_quantizers, codebook_size_with_eos, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        **kwargs
    ):
        coarse_logits, fine_logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return coarse_logits, fine_logits

        null_coarse_logits, null_fine_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_coarse_logits = None
        if exists(null_coarse_logits):
            scaled_coarse_logits =  null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        scaled_fine_logits =  null_fine_logits + (fine_logits - null_fine_logits) * cond_scale
        return scaled_coarse_logits, scaled_fine_logits

    def forward(
        self,
        coarse_token_ids,
        fine_token_ids,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_drop_prob = None,
        self_attn_mask = None,
        return_only_fine_logits = False
    ):
        b, device = coarse_token_ids.shape[0], coarse_token_ids.device
        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.embed_text(text, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids, fine_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, fine_token_ids))

        b, n = coarse_token_ids.shape

        coarse_offsets = self.codebook_size * torch.arange(self.num_coarse_quantizers, device = device)
        coarse_offsets = repeat(coarse_offsets, 'q -> 1 (n q)', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_offsets = coarse_offsets[:, :coarse_token_ids.shape[-1]]
        coarse_token_ids = coarse_token_ids + coarse_offsets

        fine_offsets = self.codebook_size * torch.arange(self.num_fine_quantizers, device = device)
        fine_offsets = repeat(fine_offsets, 'q -> 1 (n q)', n = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers))
        fine_offsets = fine_offsets[:, :fine_token_ids.shape[-1]]
        fine_token_ids = fine_token_ids + fine_offsets

        coarse_tokens = self.coarse_embedding(coarse_token_ids)
        fine_tokens = self.fine_embedding(fine_token_ids)

        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)
        fine_start_tokens = repeat(self.fine_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            coarse_start_tokens,
            coarse_tokens,
            fine_start_tokens,
            fine_tokens
        ), dim = 1)

        tokens = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask)

        pred_coarse_tokens, pred_fine_tokens = tokens[:, :n], tokens[:, (n + 1):]

        # get coarse logits

        pred_coarse_seq_len = pred_coarse_tokens.shape[1]

        padding = remainder_needed_until_multiple(pred_coarse_seq_len, self.num_coarse_quantizers)

        if padding != 0:
            pred_coarse_tokens = F.pad(pred_coarse_tokens, (0, 0, 0, padding), value = 0.)

        pred_coarse_tokens = rearrange(pred_coarse_tokens, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits = None

        if not return_only_fine_logits:
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

        return coarse_logits, fine_logits

# training wrappers

@beartype
class SemanticTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        pad_id = -1,
        unique_consecutive = True,
        mask_prob = 0.15
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.transformer = transformer
        assert not exists(self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        max_length,
        text: Optional[List[str]] = None,
        text_embeds = None,
        prime_wave = None,
        prime_ids = None,
        batch_size = 1,
        cond_scale = 3,
        filter_thres = 0.9,
        temperature = 1.,
        include_eos_in_output = True,  # if doing hierarchical sampling, eos must be kept for an easy time
        **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            ids = self.wav2vec(prime_wave, flatten = False)
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            logits = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                **kwargs
            )

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

        sample_semantic_ids = mask_out_after_eos_id(sample_semantic_ids, self.pad_id, keep_eos = False)

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

@beartype
class CoarseTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        soundstream: Optional[SoundStream]  = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        pad_id = -1,
        unique_consecutive = True,
        semantic_cross_entropy_loss_weight = 1.,
        mask_prob = 0.15
    ):
        super().__init__()
        self.soundstream = soundstream
        self.wav2vec = wav2vec

        self.transformer = transformer
        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight

        self.num_coarse_quantizers = transformer.num_coarse_quantizers
        self.semantic_eos_id = transformer.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids,
        text: Optional[List[str]] = None,
        text_embeds = None,
        max_time_steps = 512,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        **kwargs
    ):
        batch, device = semantic_token_ids.shape[0], self.device

        semantic_token_ids = semantic_token_ids.to(device)

        coarse_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # initialize

        init_coarse_time_step = coarse_token_ids.shape[-1]
        sampled_coarse_token_ids = coarse_token_ids.clone()

        for time_step in tqdm(range(init_coarse_time_step, max_time_steps), desc = 'generating coarse'):
            for ind in range(self.num_coarse_quantizers):
                is_last_step = ind == (self.num_coarse_quantizers - 1)

                _, coarse_logits = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = coarse_token_ids,
                    semantic_token_ids = semantic_token_ids,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_only_coarse_logits = True,
                    **kwargs
                )

                last_coarse_logits = coarse_logits[:, -1]

                if not is_last_step:
                    last_coarse_logits[:, -1] = float('-inf') # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval

                filtered_logits = top_k(last_coarse_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat((sampled_coarse_token_ids, sampled), dim = -1)

        sampled_coarse_token_ids = mask_out_after_eos_id(sampled_coarse_token_ids, self.coarse_eos_id, keep_eos = False)
        sampled_coarse_token_ids = rearrange(sampled_coarse_token_ids, 'b (n q) -> b n q', q = self.num_coarse_quantizers)

        if not reconstruct_wave:
            return sampled_coarse_token_ids

        assert exists(self.soundstream)

        wav = self.soundstream.decode_from_codebook_indices(sampled_coarse_token_ids)
        return rearrange(wav, 'b 1 n -> b n')

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        raw_wave_for_soundstream = None,
        coarse_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        raw_wave_for_soundstream = default(raw_wave_for_soundstream, raw_wave)
        assert exists(raw_wave_for_soundstream) or exists(coarse_token_ids), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        assert not all(map(exists, (raw_wave, raw_wave_for_soundstream, semantic_token_ids, coarse_token_ids)))

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        if not exists(coarse_token_ids):
            assert exists(self.soundstream), 'SoundStream must be provided if given raw wave for training'

            with torch.no_grad():
                self.soundstream.eval()
                _, indices, _ = self.soundstream(raw_wave_for_soundstream, return_encoded = True)
                coarse_token_ids, _ = indices[..., :self.num_coarse_quantizers], indices[..., self.num_coarse_quantizers:]

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

        semantic_logits, coarse_logits = self.transformer(
            semantic_token_ids = semantic_token_ids,
            coarse_token_ids = coarse_token_ids,
            self_attn_mask = self_attn_mask,
            **kwargs
        )

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(self_attn_mask.shape, self.mask_prob, device = self_attn_mask.device)

        # whether to early return the logits

        if not return_loss:
            return semantic_logits, coarse_logits

        coarse_logits, semantic_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), (coarse_logits, semantic_logits))

        if self.unique_consecutive:
            num_coarse_logits, num_semantic_logits = coarse_labels.numel(), (semantic_labels != self.pad_id).sum()
        else:
            num_coarse_logits, num_semantic_logits = coarse_logits.shape[-1], semantic_logits.shape[-1]

        semantic_loss = 0.
        if self.semantic_cross_entropy_loss_weight > 0:
            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index = self.pad_id
            )

        coarse_loss = F.cross_entropy(
            coarse_logits,
            coarse_labels
        )

        return (
            semantic_loss * num_semantic_logits * self.semantic_cross_entropy_loss_weight +
            coarse_loss * num_coarse_logits
        ) / (num_semantic_logits + num_coarse_logits)

@beartype
class FineTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: FineTransformer,
        soundstream: Optional[SoundStream] = None,
        coarse_cross_entropy_loss_weight = 1.,
        pad_id = -1,
        mask_prob = 0.15
    ):
        super().__init__()
        self.soundstream = soundstream
        self.transformer = transformer

        self.num_fine_quantizers = transformer.num_fine_quantizers
        self.num_coarse_quantizers = transformer.num_coarse_quantizers
        self.eos_id = transformer.eos_id

        assert self.num_coarse_quantizers > 0

        self.pad_id = pad_id
        self.coarse_cross_entropy_loss_weight = coarse_cross_entropy_loss_weight

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        coarse_token_ids,
        text: Optional[List[str]] = None,
        text_embeds = None,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
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
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # initialize

        fine_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        init_fine_time_step = fine_token_ids.shape[-1]
        max_time_steps = coarse_token_ids.shape[1] // self.num_coarse_quantizers

        sampled_fine_token_ids = fine_token_ids.clone()

        for time_step in tqdm(range(init_fine_time_step, max_time_steps), desc = 'generating fine'):
            for ind in range(self.num_fine_quantizers):
                is_last_step = ind == (self.num_fine_quantizers - 1)

                _, fine_logits = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = coarse_token_ids,
                    fine_token_ids = fine_token_ids,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_only_fine_logits = True,
                    **kwargs
                )

                last_fine_logits = fine_logits[:, -1]

                if not is_last_step:
                    last_fine_logits[:, -1] = float('-inf') # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval

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
            seq_lengths = reduce(~pos_is_all_padding, 'b n 1 -> b', 'sum')

            sampled_fine_token_ids = sampled_fine_token_ids.masked_fill(pos_is_all_padding, self.pad_id)

        # if not reconstructing wave, return just the fine token ids

        if not reconstruct_wave:
            return sampled_fine_token_ids

        # reconstruct the wave using soundstream, concatting the fine and coarse token ids together first across quantization dimension

        assert exists(self.soundstream)

        coarse_and_fine_ids = torch.cat((coarse_token_ids, sampled_fine_token_ids), dim = -1)

        wav = self.soundstream.decode_from_codebook_indices(coarse_and_fine_ids)
        return rearrange(wav, 'b 1 n -> b n')

    def forward(
        self,
        *,
        raw_wave = None,
        token_ids = None,
        coarse_token_ids = None,
        fine_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) ^ (exists(token_ids) ^ (exists(coarse_token_ids) and exists(fine_token_ids))), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        if exists(raw_wave):
            assert exists(self.soundstream), 'SoundStream must be provided if given raw wave for training'

            with torch.no_grad():
                self.soundstream.eval()
                _, token_ids, _ = self.soundstream(raw_wave, return_encoded = True)

        if exists(token_ids):
            coarse_token_ids, fine_token_ids = token_ids[..., :self.num_coarse_quantizers], token_ids[..., self.num_coarse_quantizers:]

        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        fine_token_ids = rearrange(fine_token_ids, 'b ... -> b (...)')

        if self.training:
            coarse_token_ids = append_eos_id(coarse_token_ids, self.transformer.eos_id)
            fine_token_ids = append_eos_id(fine_token_ids, self.transformer.eos_id)

        if return_loss:
            coarse_labels, fine_labels = coarse_token_ids, fine_token_ids.clone()
            fine_token_ids = fine_token_ids[:, :-1]

        # do not attend to any of the coarse padding tokens or coarse end token either

        self_attn_mask = (coarse_token_ids != self.pad_id) & (coarse_token_ids != self.eos_id)
        coarse_token_ids = coarse_token_ids.masked_fill(~self_attn_mask, 0)

        fine_token_seq_len = fine_token_ids.shape[-1]
        self_attn_mask = F.pad(self_attn_mask, (1, fine_token_seq_len + 1), value = True)

        coarse_logits, fine_logits = self.transformer(
            coarse_token_ids = coarse_token_ids,
            fine_token_ids = fine_token_ids,
            self_attn_mask = self_attn_mask,
            **kwargs
        )

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(self_attn_mask.shape, self.mask_prob, device = self_attn_mask.device)

        # early return the logits

        if not return_loss:
            return coarse_logits, fine_logits

        coarse_logits, fine_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), (coarse_logits, fine_logits))

        num_coarse_logits, num_fine_logits = coarse_logits.shape[-1], fine_logits.shape[-1]

        coarse_loss = 0.
        if self.coarse_cross_entropy_loss_weight > 0:
            coarse_loss = F.cross_entropy(
                coarse_logits,
                coarse_labels
            )

        fine_loss = F.cross_entropy(
            fine_logits,
            fine_labels
        )

        return (
            coarse_loss * num_coarse_logits * self.coarse_cross_entropy_loss_weight +
            fine_loss * num_fine_logits
        ) / (num_coarse_logits + num_fine_logits)

# audio LM

@beartype
class AudioLM(nn.Module):
    def __init__(
        self,
        *,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]], 
        soundstream: SoundStream,
        semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        unique_consecutive = True
    ):
        super().__init__()

        assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers

        self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])

        self.semantic = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = semantic_transformer,
            unique_consecutive = unique_consecutive
        )

        self.coarse = CoarseTransformerWrapper(
            wav2vec = wav2vec,
            soundstream = soundstream,
            transformer = coarse_transformer,
            unique_consecutive = unique_consecutive
        )

        self.fine = FineTransformerWrapper(
            soundstream = soundstream,
            transformer = fine_transformer
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    def forward(
        self,
        *,
        batch_size = 1,
        text: Optional[List[str]] = None,
        prime_wave = None,
        max_length = 2048,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False
    ):
        assert not (self.needs_text and not exists(text)), 'text needs to be passed in if one of the transformer requires conditioning'

        if exists(prime_wave):
            prime_wave = prime_wave.to(self.device)

        semantic_token_ids = self.semantic.generate(
            text = text if self.semantic_has_condition else None,
            batch_size = batch_size,
            prime_wave = prime_wave,
            max_length = max_length
        )

        coarse_token_ids_or_recon_wave = self.coarse.generate(
            text = text if self.coarse_has_condition else None,
            semantic_token_ids = semantic_token_ids,
            reconstruct_wave = return_coarse_generated_wave
        )

        if return_coarse_generated_wave:
            return coarse_token_ids_or_recon_wave

        generated_wave = self.fine.generate(
            text = text if self.fine_has_condition else None,
            coarse_token_ids = coarse_token_ids_or_recon_wave,
            reconstruct_wave = True,
            mask_out_generated_fine_tokens = mask_out_generated_fine_tokens
        )

        return generated_wave

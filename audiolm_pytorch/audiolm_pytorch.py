import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

from vector_quantize_pytorch import ResidualVQ

# helper functions

def exists(val):
    return val is not None

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

# discriminators

class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = 4,
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 7)
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _ in range(layers):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 8, stride = 4, padding = 4),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 3),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 1),
        )

    def forward(self, x, return_intermediates = False):
        x = self.init_conv(x)

        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

class ComplexLeakyReLU(nn.Module):
    """ just do nonlinearity on imag and real component separately for now """
    def __init__(self, p = 0.1):
        super().__init__()
        self.nonlin = leaky_relu(p)

    def forward(self, x):
        imag, real = map(self.nonlin, (x.imag, x.real))
        return torch.view_as_complex(torch.stack((imag, real), dim = -1))

def STFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        nn.Conv2d(chan_in, chan_in, 3, padding = 1, dtype = torch.complex64),
        ComplexLeakyReLU(),
        nn.Conv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings, dtype = torch.complex64)
    )

class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, channels, 7, padding = 3, dtype = torch.complex64)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = nn.ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(STFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = nn.Conv2d(layer_channels[-1], 1, (16, 1), dtype = torch.complex64) # todo: remove hardcoded 16

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')
        x = torch.view_as_complex(torch.stft(x, 256))
        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)
        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        complex_logits = self.final_conv(x)

        if not return_intermediates:
            return complex_logits

        return complex_logits, intermediates

# sound stream

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        self.causal_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)

class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7):
    return Residual(nn.Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
        nn.ELU()
    ))

def EncoderBlock(chan_in, chan_out, stride):
    return nn.Sequential(
        ResidualUnit(chan_in, chan_in, 1),
        ResidualUnit(chan_in, chan_in, 3),
        ResidualUnit(chan_in, chan_in, 9),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        ResidualUnit(chan_out, chan_out, 1),
        ResidualUnit(chan_out, chan_out, 3),
        ResidualUnit(chan_out, chan_out, 9),
    )

class SoundStream(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size = 1024,
        rq_num_quantizers = 8,
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        recon_loss_weight = 1.,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100
    ):
        super().__init__()
        self.single_channel = input_channels == 1

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3)
        )

        self.rq = ResidualVQ(
            dim = codebook_dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            kmeans_init = True,
            threshold_ema_dead_code = 2
        )

        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride))

        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7)
        )

        # discriminators

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])

        self.stft_discriminator = STFTDiscriminator()

        # loss weights

        self.recon_loss_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight

    def forward(
        self,
        x,
        return_discr_loss = False,
        return_stft_discr_loss = False
    ):
        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')
        x, indices, commit_loss = self.rq(x)
        x = rearrange(x, 'b n c -> b c n')

        recon_x = self.decoder(x)

        # stft discr loss

        if return_stft_discr_loss:
            assert self.single_channel
            real, fake = orig_x, recon_x.detach()
            stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real, fake))
            stft_discr_loss = (hinge_discr_loss(stft_fake_logits.real, stft_real_logits.real) + hinge_discr_loss(stft_fake_logits.imag, stft_real_logits.imag)) / 2
            return stft_discr_loss

        # multi-scale discriminator loss

        if return_discr_loss:
            real, fake = orig_x, recon_x.detach()
            discr_losses = []

            for discr, scale in zip(self.discriminators, self.discr_multi_scales):
                scaled_real, scaled_fake = map(lambda t: F.interpolate(t, scale_factor = scale), (real, fake))

                real_logits, fake_logits = map(discr, (scaled_real, scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)
                discr_losses.append(one_discr_loss)

            return torch.stack(discr_losses).mean()

        # recon loss

        recon_loss = F.mse_loss(orig_x, recon_x)

        # adversarial loss

        adversarial_losses = []

        discr_intermediates = []

        # adversarial loss for multi-scale discriminators

        real, fake = orig_x, recon_x

        # features from stft

        (stft_real_logits, stft_real_intermediates), (stft_fake_logits, stft_fake_intermediates) = map(partial(self.stft_discriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((stft_real_intermediates, stft_fake_intermediates))

        for discr, scale in zip(self.discriminators, self.discr_multi_scales):
            scaled_real, scaled_fake = map(lambda t: F.interpolate(t, scale_factor = scale), (real, fake))
            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(partial(discr, return_intermediates = True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)

        feature_loss = torch.stack(feature_losses).mean()

        # adversarial loss for stft discriminator

        adversarial_losses.append(hinge_gen_loss(stft_fake_logits.real))
        adversarial_losses.append(hinge_gen_loss(stft_fake_logits.imag))

        adversarial_loss = torch.stack(adversarial_losses).mean()

        return recon_loss * self.recon_loss_weight + adversarial_loss * self.adversarial_loss_weight + feature_loss * self.feature_loss_weight

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

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

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

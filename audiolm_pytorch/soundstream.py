import functools
from pathlib import Path
from functools import partial

import torch
from torch import nn, einsum
from torch.autograd import grad as torch_grad
import torch.nn.functional as F

import torchaudio.transforms as T

from einops import rearrange, reduce

from vector_quantize_pytorch import ResidualVQ

from local_attention import LocalMHA
from local_attention.transformer import FeedForward

from audiolm_pytorch.utils import curtail_to_multiple

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def gradient_penalty(wave, output, weight = 10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

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
                nn.Conv1d(curr_channels, chan_out, 8, stride = 4, padding = 4, groups = groups),
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

def STFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        nn.Conv2d(chan_in, chan_in, 3, padding = 1),
        leaky_relu(),
        nn.Conv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
        chan_mults = (1, 2, 4, 4, 8, 8),
        input_channels = 1,
        n_fft = 1024,
        hop_length = 256,
        win_length = 1024,
        stft_normalized = True
    ):
        super().__init__()
        self.stft = T.Spectrogram(
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window_fn = torch.hann_window,
            normalized = stft_normalized,
            center = False,
            pad_mode = None,
            power = None
        )

        input_channels *= 2
        self.init_conv = nn.Conv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = nn.ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(STFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = nn.Conv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:

        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        x = self.stft(x)
        x = rearrange(x, 'b ... -> b 1 ...')
        x = torch.cat((x.real, x.imag), dim = 1)

        intermediates = []

        x = self.init_conv(x)
        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        logits = self.final_conv(x)

        if not return_intermediates:
            return logits

        return logits, intermediates

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
        nn.ELU(),
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
    ))

def EncoderBlock(chan_in, chan_out, stride):
    return nn.Sequential(
        ResidualUnit(chan_in, chan_in, 1),
        ResidualUnit(chan_in, chan_in, 3),
        ResidualUnit(chan_in, chan_in, 9),
        nn.ELU(),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    return nn.Sequential(
        nn.ELU(),
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        ResidualUnit(chan_out, chan_out, 1),
        ResidualUnit(chan_out, chan_out, 3),
        ResidualUnit(chan_out, chan_out, 9),
    )

class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        **kwargs
    ):
        super().__init__()
        self.attn = LocalMHA(dim = dim, **kwargs)
        self.ff = FeedForward(dim = dim)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

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
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.95,
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        recon_loss_weight = 1.,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 24000,
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

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

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            window_size = attn_window_size,
            prenorm = True,
            causal = True
        )

        self.encoder_attn = LocalTransformerBlock(**attn_kwargs) if use_local_attn else None

        self.rq = ResidualVQ(
            dim = codebook_dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = rq_commitment_weight,
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = True,
            quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        )

        self.decoder_attn = LocalTransformerBlock(**attn_kwargs) if use_local_attn else None

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

    def decode_from_codebook_indices(self, quantized_indices):
        codes = self.rq.get_codes_from_indices(quantized_indices)
        x = reduce(codes, 'q ... -> ...', 'sum')

        x = self.decoder_attn(x) + x
        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.encoder_attn.parameters(),
            *self.decoder_attn.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        return_encoded = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False
    ):
        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        x, indices, commit_loss = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        if return_encoded:
            return x, indices, commit_loss

        recon_x = self.decoder(x)

        if return_recons_only:
            return recon_x

        # multi-scale discriminator loss

        if return_discr_loss:
            real, fake = orig_x, recon_x.detach()

            stft_discr_loss = None
            stft_grad_penalty = None
            discr_losses = []
            discr_grad_penalties = []

            if self.single_channel:
                real, fake = orig_x.clone(), recon_x.detach()
                stft_real_logits, stft_fake_logits = map(self.stft_discriminator, (real.requires_grad_(), fake))
                stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

                if apply_grad_penalty:
                    stft_grad_penalty = gradient_penalty(real, stft_discr_loss)

            for discr, scale in zip(self.discriminators, self.discr_multi_scales):
                scaled_real, scaled_fake = map(lambda t: F.interpolate(t, scale_factor = scale), (real, fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()

                if exists(stft_discr_loss):
                    all_discr_losses = all_discr_losses + stft_discr_loss

                if exists(stft_grad_penalty):
                    all_discr_losses = all_discr_losses + stft_grad_penalty

                return all_discr_losses

            # return a list of discriminator losses with List[Tuple[str, Tensor]]

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

            if exists(stft_discr_loss):
                discr_losses_pkg.append(('stft', stft_discr_loss))

            if exists(stft_grad_penalty):
                discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

            return discr_losses_pkg

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

        adversarial_losses.append(hinge_gen_loss(stft_fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()

        # sum commitment loss

        all_commitment_loss = commit_loss.sum()

        total_loss = recon_loss * self.recon_loss_weight + adversarial_loss * self.adversarial_loss_weight + feature_loss * self.feature_loss_weight + all_commitment_loss

        if return_loss_breakdown:
            return total_loss, (recon_loss, adversarial_loss, feature_loss, all_commitment_loss)

        return total_loss

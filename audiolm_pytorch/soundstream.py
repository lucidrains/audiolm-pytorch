import functools
from pathlib import Path
from functools import partial, wraps
from itertools import cycle, zip_longest
from typing import Optional, List

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.linalg import vector_norm

import torchaudio.transforms as T
from torchaudio.functional import resample

from einops import rearrange, reduce, pack, unpack

from vector_quantize_pytorch import (
    GroupedResidualVQ,
    GroupedResidualLFQ,
    GroupedResidualFSQ
)

from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias

from gateloop_transformer import SimpleGateLoopLayer as GateLoop

from audiolm_pytorch.utils import curtail_to_multiple

from audiolm_pytorch.version import __version__
from packaging import version
parsed_version = version.parse(__version__)

import pickle

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}

# gan losses

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

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
    return weight * ((vector_norm(gradients, dim = 1) - 1) ** 2).mean()

# better sequential

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# discriminators

class MultiScaleDiscriminator(Module):
    def __init__(
        self,
        channels = 16,
        layers = 4,
        groups = (4, 16, 64, 256),
        chan_max = 1024,
        input_channels = 1
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 15, padding = 7)
        self.conv_layers = ModuleList([])

        curr_channels = channels

        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 41, stride = 4, padding = 20, groups = group),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 5, padding = 2),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 3, padding = 1),
        )

    def forward(
        self,
        x,
        return_intermediates = False
    ):
        x = self.init_conv(x)
        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates

# autoregressive squeeze excitation
# https://arxiv.org/abs/1709.01507

class SqueezeExcite(Module):
    def __init__(self, dim, reduction_factor = 4, dim_minimum = 8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive

        cum_sum = x.cumsum(dim = -2)
        denom = torch.arange(1, seq + 1, device = device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate

        gate = self.net(cum_mean)

        return x * gate

# complex stft discriminator

class ModReLU(Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))

class ComplexConv2d(Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        stride = 1,
        padding = 0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype = torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride = self.stride, padding = self.padding)

def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        Residual(Sequential(
            ComplexConv2d(chan_in, chan_in, 3, padding = 1),
            ModReLU(),
            ComplexConv2d(chan_in, chan_in, 3, padding = 1)
        )),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride = strides, padding = paddings)
    )

class ComplexSTFTDiscriminator(Module):
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
        stft_normalized = False,
        stft_window_fn = torch.hann_window,
        logits_abs = True
    ):
        super().__init__()
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding = 3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1)) # todo: remove hardcoded 16

        # stft settings

        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # how to output the logits into real space

        self.logits_abs = logits_abs

    def forward(self, x, return_intermediates = False):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        stft_window = self.stft_window_fn(self.win_length, device = x.device)

        x = torch.stft(
            x,
            self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = stft_window,
            normalized = self.stft_normalized,
            return_complex = True
        )

        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)

        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        complex_logits = self.final_conv(x)

        if self.logits_abs:
            complex_logits = complex_logits.abs()
        else:
            complex_logits = torch.view_as_real(complex_logits)

        if not return_intermediates:
            return complex_logits

        return complex_logits, intermediates

# sound stream

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ChannelTranspose(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c n -> b n c')
        out = self.fn(x, **kwargs) + x
        return rearrange(out, 'b n c -> b c n')

class CausalConv1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode = 'reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode = self.pad_mode)
        return self.conv(x)

class CausalConvTranspose1d(Module):
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

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7, squeeze_excite = False, pad_mode = 'reflect'):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation, pad_mode = pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ))

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), squeeze_excite = False, pad_mode = 'reflect'):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(ResidualUnit, squeeze_excite = squeeze_excite, pad_mode = pad_mode)

    it = cycle(cycle_dilations)
    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

class LocalTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(ModuleList([
                LocalMHA(
                    dim = dim,
                    heads = heads,
                    qk_rmsnorm = True,
                    window_size = window_size,
                    use_rotary_pos_emb = not dynamic_pos_bias,
                    gate_values_per_head = True,
                    use_xpos = True,
                    **kwargs
                ),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return x

class FiLM(Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
        return x * gamma + beta

class SoundStream(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size: Optional[int] = None,
        finite_scalar_quantizer_levels: Optional[List[int]] = None,
        rq_num_quantizers = 8,
        rq_commitment_weight = 1.,
        rq_ema_decay = 0.95,
        rq_quantize_dropout_multiple_of = 1,
        rq_groups = 1,
        rq_stochastic_sample_codes = False,
        rq_kwargs: dict = {},
        use_lookup_free_quantizer = False,              # proposed in https://arxiv.org/abs/2310.05737, adapted for residual quantization
        use_finite_scalar_quantizer = False,            # proposed in https://arxiv.org/abs/2309.15505, adapted for residual quantization
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1e-5,
        adversarial_loss_weight = 1.,
        feature_loss_weight = 100,
        quantize_dropout_cutoff_index = 1,
        target_sample_hz = 16000,
        use_local_attn = True,
        attn_window_size = 128,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_depth = 1,
        attn_xpos_scale_base = None,
        attn_dynamic_pos_bias = False,
        use_gate_loop_layers = False,
        squeeze_excite = False,
        complex_stft_discr_logits_abs = True,
        pad_mode = 'reflect',
        stft_discriminator: Optional[Module] = None,  # can pass in own stft discriminator
        complex_stft_discr_kwargs: dict = dict()
    ):
        super().__init__()

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # rest of the class

        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                encoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_out, use_heinsen = False))))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7, pad_mode = pad_mode),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3, pad_mode = pad_mode)
        )

        attn_kwargs = dict(
            dim = codebook_dim,
            dim_head = attn_dim_head,
            heads = attn_heads,
            depth = attn_depth,
            window_size = attn_window_size,
            xpos_scale_base = attn_xpos_scale_base,
            dynamic_pos_bias = attn_dynamic_pos_bias,
            prenorm = True,
            causal = True
        )

        self.encoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        self.encoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.num_quantizers = rq_num_quantizers

        self.codebook_dim = codebook_dim

        self.rq_groups = rq_groups

        assert not (use_lookup_free_quantizer and use_finite_scalar_quantizer)

        self.use_lookup_free_quantizer = use_lookup_free_quantizer
        self.use_finite_scalar_quantizer = use_finite_scalar_quantizer

        if use_lookup_free_quantizer:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'

            self.rq = GroupedResidualLFQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        elif use_finite_scalar_quantizer:
            assert not exists(codebook_size) and exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to True, `finite_scalar_quantizer_levels` must be set (and not `codebook_size`). the effective codebook size is the cumulative product of all the FSQ levels'

            self.rq = GroupedResidualFSQ(
                dim = codebook_dim,
                levels = finite_scalar_quantizer_levels,
                num_quantizers = rq_num_quantizers,
                groups = rq_groups,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                **rq_kwargs
            )

            self.codebook_size = self.rq.codebook_size

        else:
            assert exists(codebook_size) and not exists(finite_scalar_quantizer_levels), 'if use_finite_scalar_quantizer is set to False, `codebook_size` must be set (and not `finite_scalar_quantizer_levels`)'
            self.rq = GroupedResidualVQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                groups = rq_groups,
                decay = rq_ema_decay,
                commitment_weight = rq_commitment_weight,
                quantize_dropout_multiple_of = rq_quantize_dropout_multiple_of,
                kmeans_init = True,
                threshold_ema_dead_code = 2,
                quantize_dropout = True,
                quantize_dropout_cutoff_index = quantize_dropout_cutoff_index,
                stochastic_sample_codes = rq_stochastic_sample_codes,
                **rq_kwargs
            )

            self.codebook_size = codebook_size

        self.decoder_film = FiLM(codebook_dim, dim_cond = 2)

        self.decoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations, squeeze_excite, pad_mode))

            if use_gate_loop_layers:
                decoder_blocks.append(Residual(ChannelTranspose(GateLoop(chan_in))))

        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7, pad_mode = pad_mode),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7, pad_mode = pad_mode)
        )

        # discriminators

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])
        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = ModuleList([nn.Identity()] + [nn.AvgPool1d(2 * factor, stride = factor, padding = factor) for factor in discr_rel_factors])

        self.stft_discriminator = stft_discriminator

        if not exists(self.stft_discriminator):
            self.stft_discriminator = ComplexSTFTDiscriminator(
                stft_normalized = stft_normalized,
                logits_abs = complex_stft_discr_logits_abs,  # whether to output as abs() or use view_as_real
                **complex_stft_discr_kwargs
            )

        # multi spectral reconstruction

        self.mel_spec_transforms = ModuleList([])
        self.mel_spec_recon_alphas = []

        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)

        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = (win_length / 2) ** 0.5

            calculated_n_fft = default(max(n_fft, win_length), win_length)  # @AndreyBocharnikov said this is usually win length, but overridable

            # if any audio experts have an opinion about these settings, please submit a PR

            melspec_transform = T.MelSpectrogram(
                sample_rate = target_sample_hz,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = stft_normalized
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)

        # loss weights

        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def configs(self):
        return pickle.loads(self._configs)

    def decode_from_codebook_indices(self, quantized_indices):
        assert quantized_indices.dtype in (torch.long, torch.int32)

        if quantized_indices.ndim == 3:
            quantized_indices = rearrange(quantized_indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        x = self.rq.get_output_from_indices(quantized_indices)

        return self.decode(x)

    def decode(self, x, quantize = False):
        if quantize:
            x, *_ = self.rq(x)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = self._configs,
            version = __version__
        )

        torch.save(pkg, str(path))

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        soundstream = cls(**config)
        soundstream.load(path, strict = strict)
        soundstream.eval()
        return soundstream

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # check version

        if 'version' in pkg and version.parse(pkg['version']) < parsed_version:
            print(f'soundstream model being loaded was trained on an older version of audiolm-pytorch ({pkg["version"]})')

        has_ema = 'ema_model' in pkg
        model_pkg = pkg['ema_model'] if has_ema else pkg['model']

        if has_ema:
            model_pkg = filter_by_keys(lambda k: k.startswith('ema_model.'), model_pkg)
            model_pkg = map_keys(lambda k: k[len('ema_model.'):], model_pkg)

        self.load_state_dict(model_pkg, strict = strict)

    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *(self.encoder_attn.parameters() if exists(self.encoder_attn) else []),
            *(self.decoder_attn.parameters() if exists(self.decoder_attn) else []),
            *self.encoder_film.parameters(),
            *self.decoder_film.parameters(),
            *self.rq.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def process_input(
        self,
        x,
        input_sample_hz = None,
        curtail_from_left = False
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of, from_left = curtail_from_left)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        return x, ps

    @torch.no_grad()
    def tokenize(self, audio):
        self.eval()
        return self.forward(audio, return_codes_only = True)

    def forward(
        self,
        x,
        target = None,
        is_denoising = None, # if you want to learn film conditioners that teach the soundstream to denoise - target would need to be passed in above
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_loss_breakdown = False,
        return_recons_only = False,
        input_sample_hz = None,
        apply_grad_penalty = False,
        curtail_from_left = False
    ):
        assert not (exists(is_denoising) and not exists(target))

        process_input = partial(self.process_input, input_sample_hz = input_sample_hz, curtail_from_left = curtail_from_left)

        x, ps = process_input(x)

        if exists(target):
            target, _ = process_input(target)

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype, device = self.device) # [1, 0] for denoise, [0, 1] for not denoising
            x = self.encoder_film(x, denoise_input)

        if not self.use_finite_scalar_quantizer:
            x, indices, commit_loss = self.rq(x)
        else:
            # finite scalar quantizer does not have any aux loss

            x, indices = self.rq(x)
            commit_loss = self.zero

        if return_codes_only:
            return indices

        if return_encoded:
            indices = rearrange(indices, 'g b n q -> b n (g q)')
            return x, indices, commit_loss

        if exists(is_denoising):
            x = self.decoder_film(x, denoise_input)

        if exists(self.decoder_attn):
            x = self.decoder_attn(x)

        x = rearrange(x, 'b n c -> b c n')

        recon_x = self.decoder(x)

        if return_recons_only:
            recon_x, = unpack(recon_x, ps, '* c n')
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

            scaled_real, scaled_fake = real, fake
            for discr, downsample in zip(self.discriminators, self.downsamples):
                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
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

        target = default(target, orig_x)  # target can also be passed in, in the case of denoising

        recon_loss = F.mse_loss(target, recon_x)

        # multispectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312

        multi_spectral_recon_loss = self.zero

        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (orig_x, recon_x))
                log_orig_mel, log_recon_mel = map(log, (orig_mel, recon_mel))

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim = -2).mean()
                l2_log_mel_loss = alpha * vector_norm(log_orig_mel - log_recon_mel, dim = -2).mean()

                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss

        # adversarial loss

        adversarial_losses = []

        discr_intermediates = []

        # adversarial loss for multi-scale discriminators

        real, fake = orig_x, recon_x

        # features from stft

        (stft_real_logits, stft_real_intermediates), (stft_fake_logits, stft_fake_intermediates) = map(partial(self.stft_discriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((stft_real_intermediates, stft_fake_intermediates))

        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

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

        total_loss = recon_loss * self.recon_loss_weight + multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + adversarial_loss * self.adversarial_loss_weight + feature_loss * self.feature_loss_weight + all_commitment_loss

        if return_loss_breakdown:
            return total_loss, (recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss, all_commitment_loss)

        return total_loss

# some default soundstreams

def AudioLMSoundStream(
    strides = (2, 4, 5, 8),
    target_sample_hz = 16000,
    rq_num_quantizers = 12,
    **kwargs
):
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )

def MusicLMSoundStream(
    strides = (3, 4, 5, 8),
    target_sample_hz = 24000,
    rq_num_quantizers = 12,
    **kwargs
):
    return SoundStream(
        strides = strides,
        target_sample_hz = target_sample_hz,
        rq_num_quantizers = rq_num_quantizers,
        **kwargs
    )

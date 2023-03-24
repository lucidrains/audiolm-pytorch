from torch import nn
from encodec import EncodecModel
from encodec.utils import convert_audio, _linear_overlap_add

class EncodecWrapper(nn.Module):
    """
    Support pretrained 24kHz Encodec by Meta AI, if you want to skip training SoundStream.

    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    - see if we can use the 48kHz model, which is specifically for music. Right now we're using the 24kHz model because
        that's what was used in MusicLM and avoids any resampling issues.
    -

    """
    def __init__(self,
                 target_sample_hz=24000,
                 strides=(2,4,5,8),
                 num_quantizers=8,
                 ):
        super().__init__()
        # Instantiate a pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False # this means we don't need to scale codes e.g. when running model.encode(wav)

        # bandwidth affects num quantizers used: https://github.com/facebookresearch/encodec/pull/41
        self.model.set_target_bandwidth(6.0)
        assert num_quantizers == 8, "assuming 8 quantizers for now, see bandwidth comment above"

        # Fields that SoundStream has that get used externally. We replicate them here.
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"
        self.num_quantizers = num_quantizers
        self.strides = strides # used in seq_len_multiple_of

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(self, x, x_sampling_rate=24000, **kwargs):
        # kwargs for stuff like return_encoded=True, which SoundStream uses but Encodec doesn't
        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."
        # convert_audio up-samples if necessary, e.g. if wav has n samples at 16 kHz and model is 48 kHz,
        # then resulting wav has 3n samples because you do n * 48/16
        # Note: this is a bit of a hack but we avoid any resampling issues here if we just try 24kHz throughout
        # which makes convert_audio a no-op
        wav = convert_audio(x, x_sampling_rate, self.model.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [batch, num_quantizers, timesteps]
        # in original soundstream, is x, indices, commit_loss. But we only use indices in eval mode, so just keep that.
        return None, codes, None

    def decode_from_codebook_indices(self, quantized_indices):
        # Input: batch x num tokens x num quantizers
        # Output: batch x 1 x num samples

        assert self.model.sample_rate == 24000,\
            "if changing to 48kHz, that model segments its audio into lengths of 1.0 second with 1% overlap, whereas " \
            "the 24kHz doesn't segment at all. this means the frame decode logic might change; this is a reminder to " \
            "double check that."
        # Since 24kHz pretrained doesn't do any segmenting, we have all the frames already (1 frame = 1 token in quantized_indices)

        # The following code is hacked in from self.model.decode() (Encodec version 0.1.1) where we skip the part about
        # scaling.
        # Shape: 1 x (num_frames * stride product). 1 because we have 1 frame (because no segmenting)
        frames = self._decode_frame(quantized_indices)
        result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        # TODO: I'm not overly pleased with this because when this function gets called, we just rearrange the result
        #   back to b n anyways, but we'll keep this as a temporary hack just to make things work for now
        return rearrange(result, 'b n -> b 1 n')

    def _decode_frame(self, quantized_indices):
        # The following code is hacked in from self.model._decode_frame() (Encodec version 0.1.1) where we assume we've
        # already unwrapped the EncodedFrame
        # Input: batch x num tokens x num quantizers
        # Output: batch x new_num_samples, where new_num_samples is num_frames * stride product (may be slightly
        # larger than original num samples as a result, because the last frame might not be "fully filled" with samples
        # if num_samples doesn't divide perfectly).
        # num_frames == the number of acoustic tokens you have, one token per frame
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        # emb shape: batch x self.model.quantizer.dimension x T. Note self.model.quantizer.dimension is the embedding dimension
        return self.model.decoder(emb)

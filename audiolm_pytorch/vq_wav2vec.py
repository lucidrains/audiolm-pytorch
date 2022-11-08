from pathlib import Path

import torch
from torch import nn
from einops import rearrange

import fairseq

class FairseqVQWav2Vec(nn.Module):
    def __init__(
        self,
        checkpoint_path
    ):
        super().__init__()
        path = Path(checkpoint_path)
        assert path.exists(), f'path {checkpoint_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.eval()

    @property
    def groups(self):
        return self.model.vector_quantizer.groups

    @property
    def codebook_size(self):
        return self.model.vector_quantizer.embedding.shape[0]

    @torch.no_grad()
    def forward(self, wav_input, flatten = True):
        embed = self.model.feature_extractor(wav_input)
        _, codebook_indices = self.model.vector_quantizer.forward_idx(embed)

        if not flatten:
            return codebook_indices

        return rearrange(codebook_indices, 'b ... -> b (...)')

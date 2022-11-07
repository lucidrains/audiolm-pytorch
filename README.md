<img src="./audiolm.png" width="600px"></img>

## AudioLM - Pytorch (wip)

Implementation of <a href="https://google-research.github.io/seanet/audiolm/examples/">AudioLM</a>, a Language Modeling Approach to Audio Generation out of Google Research, in Pytorch

## Install

```bash
$ pip install audiolm-pytorch
```

## Usage

```python
import torch
from audiolm_pytorch.audiolm_pytorch import SoundStream, AudioLM, FineTransformer, FineTransformerWrapper

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

train_wrapper = FineTransformerWrapper(
    soundstream = soundstream,
    transformer = transformer
).cuda()

raw_waveform = torch.randn(1, 320 * 512).cuda()

loss = train_wrapper(
    raw_wave = raw_waveform,
    return_loss = True
)

loss.backward()
```

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing accelerate library

- <a href="https://ai.facebook.com/">MetaAI</a> for <a href="https://github.com/facebookresearch/fairseq">Fairseq</a> and the liberal license

## Todo

- [x] complete CoarseTransformer

- [ ] complete full training code for soundstream, taking care of discriminator training
- [ ] use huggingface wav2vec for embeddings, use VQ library for learning the kmeans through reconstruction task
- [ ] figure out how to do the normalization across each dimension mentioned in the paper, but ignore it for v1 of the framework
- [ ] complete sampling code for both Coarse and Fine Transformers, which will be tricky
- [ ] accommodate variable lengthed audio, bring in eos token
- [ ] full transformer training code for all three transformers
- [ ] make sure full inference with or without prompting works on the `AudioLM` class
- [ ] offer option to weight tie coarse, fine, and semantic embeddings across the 3 hierarchical transformers
- [ ] DRY a little at the end

## Citations

```bibtex
@inproceedings{Borsos2022AudioLMAL,
  title  = {AudioLM: a Language Modeling Approach to Audio Generation},
  author = {Zal{\'a}n Borsos and Rapha{\"e}l Marinier and Damien Vincent and Eugene Kharitonov and Olivier Pietquin and Matthew Sharifi and Olivier Teboul and David Grangier and Marco Tagliasacchi and Neil Zeghidour},
  year   = {2022}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2107.03312,
  title  = {SoundStream: An End-to-End Neural Audio Codec},
  author = {Zeghidour, Neil and Luebs, Alejandro and Omran, Ahmed and Skoglund, Jan and Tagliasacchi, Marco},
  publisher = {arXiv},
  url    = {https://arxiv.org/abs/2107.03312},
  year   = {2021}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

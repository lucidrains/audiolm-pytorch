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

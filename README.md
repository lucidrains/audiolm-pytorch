<img src="./audiolm.png" width="600px"></img>

## AudioLM - Pytorch (wip)

Implementation of <a href="https://google-research.github.io/seanet/audiolm/examples/">AudioLM</a>, a Language Modeling Approach to Audio Generation out of Google Research, in Pytorch

It also extends the work for conditioning with classifier free guidance with T5. This allows for one to do text-to-audio or TTS, not offered in the paper.

## Install

```bash
$ pip install audiolm-pytorch
```

## Usage

First, `SoundStream` needs to be trained on a large corpus of audio data

```python
from audiolm_pytorch import SoundStream, SoundStreamTrainer

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = '/path/to/librispeech',
    batch_size = 4,
    data_max_length = 320 * 32,
    num_train_steps = 10000
).cuda()

trainer.train()
```

Then three separate transformers (`SemanticTransformer`, `CoarseTransformer`, `FineTransformer`) need to be trained


ex. `SemanticTransformer`

```python
import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    wav2vec = wav2vec,
    dim = 1024,
    depth = 6
).cuda()

wave = torch.randn(1, 320 * 512).cuda()

loss = semantic_transformer(
    raw_wave = wave,
    return_loss = True
)

loss.backward()

# after much training above

sample = semantic_transformer.generate(max_length = 128) # (1, < 128) - may terminate early if it detects [eos]
```

ex. `CoarseTransformer`

```python
import torch
from audiolm_pytorch import HubertWithKmeans, SoundStream, CoarseTransformer, CoarseTransformerWrapper

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

coarse_transformer = CoarseTransformer(
    wav2vec = wav2vec,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

coarse_wrapper = CoarseTransformerWrapper(
    wav2vec = wav2vec,
    soundstream = soundstream,
    transformer = coarse_transformer
).cuda()

wave = torch.randn(1, 32 * 320).cuda()

loss = coarse_wrapper(
    raw_wave = wave,
    return_loss = True
)

loss.backward()

# after a lot of training

mock_semantic_token_ids = torch.randint(0, wav2vec.codebook_size, (1, 128))

coarse_tokens = coarse_wrapper.generate(
    semantic_token_ids = mock_semantic_token_ids,
    max_time_steps = 512
) # (1, 512, 3) - (batch, time steps, num quantizers)

```

ex. `FineTransformer`

```python
import torch
from audiolm_pytorch import SoundStream, FineTransformer, FineTransformerWrapper

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load('/path/to/trained/soundstream.pt')

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

wave = torch.randn(1, 320 * 512).cuda()

loss = train_wrapper(
    raw_wave = wave,
    return_loss = True
)

loss.backward()
```

All together now

```python

audiolm = AudioLM(
    wav2vec = wav2vec,
    soundstream = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = transformer
)

generated_wav = audiolm(batch_size = 1)

# or with priming

generated_wav_with_prime = audiolm(prime_wave = torch.randn(1, 320 * 8))

# or with text condition, if given

generated_wav_with_text_condition = audiolm(text = ['chirping of birds and the distant echos of bells'])

```

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing accelerate and transformers libraries

- <a href="https://ai.facebook.com/">MetaAI</a> for <a href="https://github.com/facebookresearch/fairseq">Fairseq</a> and the liberal license

- <a href="https://github.com/eonglints">@eonglints</a> for offering his professional advice and expertise

## Todo

- [x] complete CoarseTransformer
- [x] use fairseq vq-wav2vec for embeddings
- [x] add conditioning
- [x] add classifier free guidance
- [x] add unique consecutive for 
- [x] incorporate ability to use hubert intermediate features as semantic tokens, recommended by <a href="https://github.com/lucidrains/audiolm-pytorch/discussions/13">eonglints</a>
- [x] accommodate variable lengthed audio, bring in eos token
- [x] make sure unique consecutive works with coarse transformer
- [x] pretty printing all discriminator losses to log
- [x] handle when generating semantic tokens, that last logits may not be necessarily the last in the sequence given unique consecutive processing

- [ ] complete full training code for soundstream, taking care of discriminator training
- [ ] figure out how to do the normalization across each dimension mentioned in the paper, but ignore it for v1 of the framework
- [ ] complete sampling code for both Coarse and Fine Transformers, which will be tricky
- [ ] full transformer training code for all three transformers
- [ ] wire up sample hz from sound dataset -> transformers, and have proper resampling within during training - think about whether to allow for dataset to have sound files of varying or enforce same sample hz
- [ ] make sure full inference with or without prompting works on the `AudioLM` class
- [ ] offer option to weight tie coarse, fine, and semantic embeddings across the 3 hierarchical transformers
- [ ] DRY a little at the end
- [ ] figure out how to suppress logging in fairseq
- [ ] test with speech synthesis for starters
- [ ] abstract out conditioning + classifier free guidance into external module or potentially a package
- [ ] add option to use flash attention
- [ ] simplify training even more within AudioLM class
- [ ] cli tool, something like `audiolm generate <wav.file | text>` and save generated wav file to local directory
- [ ] refactor so semantic transformer has a wrapper to that handles unique consecutives as well as wav to hubert or vq-wav2vec
- [ ] validation function within audiolm that ensures all the pieces are compatible

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

```bibtex
@article{Shazeer2019FastTD,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam M. Shazeer},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1911.02150}
}
```

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@misc{crowson2022,
    author  = {Katherine Crowson},
    url     = {https://twitter.com/rivershavewings}
}
```

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

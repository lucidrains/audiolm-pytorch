from setuptools import setup, find_packages
exec(open('audiolm_pytorch/version.py').read())

setup(
  name = 'audiolm-pytorch',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'AudioLM - Language Modeling Approach to Audio Generation from Google Research - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/audiolm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation'
  ],
  install_requires=[
    'accelerate>=0.24.0',
    'beartype>=0.16.1',
    'einops>=0.7.0',
    'ema-pytorch>=0.2.2',
    'encodec',
    'fairseq',
    'wandb',
    'gateloop-transformer>=0.2.3',
    'joblib',
    'local-attention>=1.9.0',
    'pytorch-warmup',
    'scikit-learn',
    'sentencepiece',
    'torch>=2.1',
    'torchaudio',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch>=1.18.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

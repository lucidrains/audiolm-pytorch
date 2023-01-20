from setuptools import setup, find_packages

setup(
  name = 'audiolm-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.7.1',
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
    'accelerate',
    'beartype',
    'einops>=0.6',
    'ema-pytorch',
    'fairseq',
    'joblib',
    'local-attention>=1.5.7',
    'Mega-pytorch',
    'scikit-learn',
    'sentencepiece',
    'torch>=1.6',
    'torchaudio',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch>=0.10.15'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

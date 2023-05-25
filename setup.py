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
    'accelerate',
    'beartype',
    'einops>=0.6.1',
    'ema-pytorch>=0.2.2',
    'encodec',
    'fairseq',
    'joblib',
    'lion-pytorch',
    'local-attention>=1.8.4',
    'scikit-learn',
    'sentencepiece',
    'torch>=1.12',
    'torchaudio',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch>=1.5.14'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

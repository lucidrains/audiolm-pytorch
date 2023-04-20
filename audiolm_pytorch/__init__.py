import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from audiolm_pytorch.audiolm_pytorch import AudioLM
from audiolm_pytorch.soundstream import SoundStream, AudioLMSoundStream, MusicLMSoundStream
from audiolm_pytorch.encodec import EncodecWrapper

from audiolm_pytorch.audiolm_pytorch import SemanticTransformer, CoarseTransformer, FineTransformer
from audiolm_pytorch.audiolm_pytorch import FineTransformerWrapper, CoarseTransformerWrapper, SemanticTransformerWrapper

from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans

from audiolm_pytorch.trainer import SoundStreamTrainer, SemanticTransformerTrainer, FineTransformerTrainer, CoarseTransformerTrainer

from audiolm_pytorch.audiolm_pytorch import get_embeds

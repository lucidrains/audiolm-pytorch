import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import data
from torch.utils.data import Dataset, DataLoader
from itertools import islice
import tarfile
import json
import torchaudio
import io
import glob
import soundfile as sf


wav2vec = HubertWithKmeans(
    checkpoint_path = '/fsx/home-marianna/cc_wat/audiolm-pytorch/hubert/hubert_base_ls960.pt',
    kmeans_path = '/fsx/home-marianna/cc_wat/audiolm-pytorch/hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = 500,
    dim = 1024,
    depth = 6,
    has_condition = True,               # this will have to be set to True
    cond_as_self_attn_prefix = True     # whether to condition as prefix to self attention, instead of cross attention, as was done in 'VALL-E' paper
).cuda()

dataset = data.get_dataset([f's3://s-laion/CC_AUDIO_WDS/0/{i:05d}.tar' for i in range(100)])

valid_dataset = data.get_dataset([f's3://s-laion/CC_AUDIO_WDS/0/{i:05d}.tar' for i in range(100, 124)])

trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    dataset = dataset,
    valid_dataset = valid_dataset,
    batch_size = 32,
    num_workers = 8,
    grad_accum_every = 8,
    data_max_length = 320 * 32,
    num_train_steps = 1_000_000
)

trainer.train()
from pathlib import Path
from functools import partial
import soundfile as sf

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def exists(val):
    return val is not None

# dataset functions

class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav'],
        seq_len_multiple_of = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.seq_len_multiple_of = seq_len_multiple_of

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data, _ = sf.read(file)

        if exists(self.seq_len_multiple_of):
            mult = self.seq_len_multiple_of
            data_len = len(data)
            data = data[:(data_len // mult * mult)]

        return torch.from_numpy(data).float()

# dataloader functions

def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = partial(pad_sequence, batch_first = True) if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

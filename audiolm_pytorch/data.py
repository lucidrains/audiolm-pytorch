from pathlib import Path
from functools import partial, wraps

from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

import torchaudio
from torchaudio.functional import resample

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from audiolm_pytorch.utils import curtail_to_multiple

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# dataset functions

@beartype
class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav'],
        max_length = None,
        target_sample_hz: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
        seq_len_multiple_of = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.max_length = max_length

        self.target_sample_hz = cast_tuple(target_sample_hz)
        self.seq_len_multiple_of = seq_len_multiple_of

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)
        
        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        # resample if target_sample_hz is not None in the tuple

        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually

        for data in data_tuple:
            audio_length = data.size(1)

            # pad or curtail

            if audio_length > self.max_length:
                max_start = audio_length - self.max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + self.max_length]

            else:
                data = F.pad(data, (0, self.max_length - audio_length), 'constant')

            data = rearrange(data, '1 ... -> ...')

            if exists(self.max_length):
                data = data[:self.max_length]

            if exists(self.seq_len_multiple_of):
                data = curtail_to_multiple(data, self.seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

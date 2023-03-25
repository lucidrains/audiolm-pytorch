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

import webdataset as wds
import json

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]

# dataset functions

def preprocess(sample:tuple, target_sample_hz:int=16000, max_length:int=320 * 32, seq_len_multiple_of=None):
    '''
    Resamples audio to target_sr and selects random seq_len seconds segment from audio
    if audio is shorter than seq_len, repeats audio k times (k=seq_len/audio_len, where audio_len lenght of audio)
    Converts all audio samples to mono format
    Converts captions from JSON to string:
        if there's audio meta tags like title, artist, genre constructs caption playing {genre} song "{title}" by {artist}
        uses raw cpation otherwise
    '''
    audio, json_data = sample
    label = f'{json_data["caption"]}'
   
    audio_meta = json_data.get('audio_meta', None)
    
    if audio_meta is not None:
        tags = audio_meta.get('tags', None)
        if tags is not None:
            try:
                title, artist, genre = '', '', ''
                for k in tags.keys():
                    if k in ['title', 'TITLE']:
                        title = f'titled {tags[k]}'
                    if k in ['artist', 'ARTIST']:
                        artist = f'by {tags[k]}'
                    if k in ['genre', 'GENRE']:
                        genre = tags[k]

                label = f'playing {genre} song "{title}" {artist}'
            except:
                pass
    data, sample_hz = audio
    num_outputs = len(cast_tuple(target_sample_hz))
    max_length = cast_tuple(max_length, num_outputs)
    seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)
    data = cast_tuple(data, num_outputs)

    # resample if target_sample_hz is not None in the tuple

    data_tuple = tuple((resample(d, sample_hz, target_sample) if exists(target_sample) else d) for d, target_sample in zip(data, cast_tuple(target_sample_hz)))

    output = []

    # process each of the data resample at different frequencies individually

    for data, max_length_, seq_len_multiple_of_ in zip(data_tuple, max_length, seq_len_multiple_of):
        audio_length = data.size(1)

        # pad or curtail

        if audio_length > max_length_:
            max_start = audio_length - max_length_
            start = torch.randint(0, max_start, (1, ))
            data = data[:, start:start + max_length_]

        else:
            data = F.pad(data, (0, max_length_ - audio_length), 'constant')

        data = torch.mean(data, dim=0).unsqueeze(0)
        data = rearrange(data, '1 ... -> ...')

        if exists(max_length_):
            data = data[:max_length_]

        if exists(seq_len_multiple_of_):
            data = curtail_to_multiple(data, seq_len_multiple_of_)

        output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return label, output[0]

    return label, output

def get_dataset(urls: list):
    '''
    Pass s3 urls and get processed torch dataset
    '''
    urls = [f'pipe:aws s3 cp {url} -' for url in urls]
    dataset = (
           wds.WebDataset(urls)
           .decode(wds.torch_audio)
           .to_tuple("flac", "json")
           .map(preprocess)
    )
    return dataset



@beartype
class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length = cast_tuple(max_length, num_outputs)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        # resample if target_sample_hz is not None in the tuple

        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            # pad or curtail

            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]

            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

            data = rearrange(data, '1 ... -> ...')

            if exists(max_length):
                data = data[:max_length]

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

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

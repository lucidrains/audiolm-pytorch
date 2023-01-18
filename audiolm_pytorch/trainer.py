from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree

from beartype.typing import Union, List, Optional, Tuple
from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from einops import rearrange

from audiolm_pytorch.optimizer import get_optimizer

from ema_pytorch import EMA

from audiolm_pytorch.soundstream import SoundStream

from audiolm_pytorch.audiolm_pytorch import (
    SemanticTransformer,
    SemanticTransformerWrapper,
    CoarseTransformer,
    CoarseTransformerWrapper,
    FineTransformer,
    FineTransformerWrapper,
    FairseqVQWav2Vec,
    HubertWithKmeans
)

from audiolm_pytorch.data import SoundDataset, get_dataloader

from accelerate import Accelerator

# constants

DEFAULT_SAMPLE_RATE = 16000

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    text = List[str],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

# main trainer class

class SoundStreamTrainer(nn.Module):
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        num_train_steps,
        batch_size,
        data_max_length = None,
        folder,
        lr = 3e-4,
        grad_accum_every = 4,
        wd = 0.,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        ema_beta = 0.995,
        ema_update_after_step = 500,
        ema_update_every = 10,
        apply_grad_penalty_every = 4,
        dl_num_workers = 0,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.soundstream = soundstream
        self.ema_soundstream = EMA(soundstream, beta = ema_beta, update_after_step = ema_update_after_step, update_every = ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(soundstream.non_discr_parameters(), lr = lr, wd = wd)

        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = get_optimizer(discr.parameters(), lr = lr, wd = wd)
            setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

        self.discr_optim = get_optimizer(soundstream.stft_discriminator.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        self.ds = SoundDataset(
            folder,
            max_length = data_max_length,
            target_sample_hz = soundstream.target_sample_hz,
            seq_len_multiple_of = soundstream.seq_len_multiple_of
        )

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

        # prepare with accelerator

        (
            self.soundstream,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.soundstream,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        )

        # prepare the multiscale discriminators with accelerator

        for name, _ in self.multiscale_discriminator_iter():
            optimizer = getattr(self, name)
            optimizer = self.accelerator.prepare(optimizer)
            setattr(self, name, optimizer)

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.accelerator.init_trackers("soundstream", config=hps)        

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.soundstream),
            ema_model = self.ema_soundstream.state_dict(),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.state_dict()

        torch.save(pkg, path)

    @property
    def unwrapped_soundstream(self):
        return self.accelerator.unwrap_model(self.soundstream)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        self.unwrapped_soundstream.load_state_dict(pkg['model'])

        self.ema_soundstream.load_state_dict(pkg['ema_model'])
        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            discr_optim.load_state_dict(pkg[key])

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.unwrapped_soundstream.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    def multiscale_discriminator_optim_iter(self):
        for name, _ in self.multiscale_discriminator_iter():
            yield name, getattr(self, name)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.soundstream.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            wave, = next(self.dl_iter)
            wave = wave.to(device)

            loss, (recon_loss, *_) = self.soundstream(wave, return_loss_breakdown = True)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, dict(
                loss = loss.item() / self.grad_accum_every,
                recon_loss = recon_loss.item() / self.grad_accum_every
            ))

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        self.discr_optim.zero_grad()

        for name, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
            multiscale_discr_optim.zero_grad()

        for _ in range(self.grad_accum_every):
            wave, = next(self.dl_iter)
            wave = wave.to(device)

            discr_losses = self.soundstream(
                wave,
                apply_grad_penalty = apply_grad_penalty,
                return_discr_loss = True,
                return_discr_losses_separately = True
            )

            for name, discr_loss in discr_losses:
                self.accelerator.backward(discr_loss / self.grad_accum_every, retain_graph = True)
                accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

        if exists(self.discr_max_grad_norm):
            self.accelerator.clip_grad_norm_(self.soundstream.stft_discriminator.parameters(), self.discr_max_grad_norm)

        # gradient step for all discriminators

        self.discr_optim.step()

        for name, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
            multiscale_discr_optim.step()

        # build pretty printed losses

        losses_str = f"{steps}: soundstream total loss: {logs['loss']:.3f}, soundstream recon loss: {logs['recon_loss']:.3f}"
        self.accelerator.log({"total_loss": logs['loss'], "recon_loss": logs['recon_loss']}, step=steps)

        for key, loss in logs.items():
            if not key.startswith('scale:'):
                continue
            _, scale_factor = key.split(':')

            losses_str += f" | discr (scale {scale_factor}) loss: {loss:.3f}"
            self.accelerator.log({f"discr_loss (scale {scale_factor})": loss}, step=steps)

        # log

        self.print(losses_str)

        # update exponential moving averaged generator

        if self.is_main:
            self.ema_soundstream.update()

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            for model, filename in ((self.ema_soundstream.ema_model, f'{steps}.ema'), (self.soundstream, str(steps))):
                model.eval()

                wave, = next(self.valid_dl_iter)
                wave = wave.to(device)

                recons = model(wave, return_recons_only = True)

                milestone = steps // self.save_results_every

                for ind, recon in enumerate(recons.unbind(dim = 0)):
                    filename = str(self.results_folder / f'sample_{steps}.flac')
                    torchaudio.save(filename, recon.cpu().detach(), DEFAULT_SAMPLE_RATE)

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.soundstream.state_dict()
            model_path = str(self.results_folder / f'soundstream.{steps}.pt')
            torch.save(state_dict, model_path)

            ema_state_dict = self.ema_soundstream.state_dict()
            model_path = str(self.results_folder / f'soundstream.{steps}.ema.pt')
            torch.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

# semantic transformer trainer

@beartype
class SemanticTransformerTrainer(nn.Module):
    def __init__(
        self,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
        transformer: SemanticTransformer,
        *,
        num_train_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.wav2vec = wav2vec
        self.transformer = transformer

        self.train_wrapper = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = transformer
        )

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset
        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = wav2vec.target_sample_hz,
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
            )

        self.ds_fields = None

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True)

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every    

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.accelerator.init_trackers("semantic", config=hps)

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.ds_fields, data))

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            loss = self.train_wrapper(**data_kwargs, return_loss = True)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))

            with torch.no_grad():
                self.train_wrapper.eval()
                valid_loss = self.train_wrapper(**data_kwargs, return_loss = True)

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.transformer.state_dict()
            model_path = str(self.results_folder / f'semantic.transformer.{steps}.pt')
            torch.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

# fine transformer trainer

@beartype
class CoarseTransformerTrainer(nn.Module):
    def __init__(
        self,
        transformer: CoarseTransformer,
        soundstream: SoundStream,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
        *,
        num_train_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        ds_fields: Tuple[str, ...] = ('raw_wave', 'raw_wave_for_soundstream', 'text'),
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.transformer = transformer
        self.soundstream = soundstream
        self.wav2vec = wav2vec

        self.train_wrapper = CoarseTransformerWrapper(
            soundstream = soundstream,
            wav2vec = wav2vec,
            transformer = transformer
        )

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = (
                    wav2vec.target_sample_hz,
                    soundstream.target_sample_hz
                ), # need 2 waves resampled differently here
                seq_len_multiple_of = soundstream.seq_len_multiple_of
            )

        self.ds_fields = ds_fields

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True)

        # prepare with accelerator

        (
            self.transformer,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.transformer,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every    

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.accelerator.init_trackers("coarse", config=hps)        

        self.train_wrapper.to(self.device)

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))

            loss = self.train_wrapper(
                **data_kwargs,
                return_loss = True
            )

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))

            with torch.no_grad():
                self.train_wrapper.eval()

                valid_loss = self.train_wrapper(
                    **data_kwargs,
                    return_loss = True
                )

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.transformer.state_dict()
            model_path = str(self.results_folder / f'fine.transformer.{steps}.pt')
            torch.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

# fine transformer trainer

@beartype
class FineTransformerTrainer(nn.Module):
    def __init__(
        self,
        transformer: FineTransformer,
        soundstream: SoundStream,
        *,
        num_train_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.transformer = transformer
        self.soundstream = soundstream

        self.train_wrapper = FineTransformerWrapper(
            soundstream = soundstream,
            transformer = transformer
        )

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        if not exists(self.ds):
            assert exists(folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length = data_max_length,
                target_sample_hz = soundstream.target_sample_hz,
                seq_len_multiple_of = soundstream.seq_len_multiple_of
            )

        self.ds_fields = None

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True)

        # prepare with accelerator

        (
            self.transformer,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.transformer,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every    

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.accelerator.init_trackers("fine", config=hps)        

        self.train_wrapper.to(self.device)

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.ds_fields, data))

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))
            loss = self.train_wrapper(**data_kwargs, return_loss = True)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.valid_dl_iter))

            with torch.no_grad():
                self.train_wrapper.eval()
                valid_loss = self.train_wrapper(**data_kwargs, return_loss = True)

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.transformer.state_dict()
            model_path = str(self.results_folder / f'fine.transformer.{steps}.pt')
            torch.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

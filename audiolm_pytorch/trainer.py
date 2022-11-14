from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree
from PIL import Image

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from einops import rearrange

from audiolm_pytorch.optimizer import get_optimizer

from ema_pytorch import EMA

from audiolm_pytorch.audiolm_pytorch import SoundStream
from audiolm_pytorch.data import SoundDataset, get_dataloader

from accelerate import Accelerator

# constants

DEFAULT_SAMPLE_RATE = 16000

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

# main trainer class

class SoundStreamTrainer(nn.Module):
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        num_train_steps,
        batch_size,
        folder,
        lr = 3e-4,
        grad_accum_every = 1,
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
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.soundstream = soundstream
        self.ema_soundstream = EMA(soundstream, update_after_step = ema_update_after_step, update_every = ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(soundstream.parameters())
        discr_parameters = set(soundstream.stft_discriminator.parameters())
        soundstream_parameters = all_parameters - discr_parameters

        self.soundstream_parameters = soundstream_parameters

        self.optim = get_optimizer(soundstream_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        self.ds = SoundDataset(
            folder,
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

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True)

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

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

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
            wave = next(self.dl_iter)
            wave = wave.to(device)

            loss = self.soundstream(wave)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(self.soundstream.stft_discriminator):
            for _ in range(self.grad_accum_every):
                wave = next(self.dl_iter)
                wave = wave.to(device)

                loss = self.soundstream(wave, return_discr_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(self.soundstream.stft_discriminator.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()
            self.discr_optim.zero_grad()

            # log

            self.print(f"{steps}: soundstream loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        # update exponential moving averaged generator

        if self.is_main:
            self.ema_soundstream.update()

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            for model, filename in ((self.ema_soundstream.ema_model, f'{steps}.ema'), (self.soundstream, str(steps))):
                model.eval()

                wave = next(self.valid_dl_iter)
                wave = wave.to(device)

                recons = model(wave, return_recons_only = True)

                milestone = steps // self.save_results_every

                for ind, recon in enumerate(recons.unbind(dim = 0)):
                    filename = str(self.results_folder / f'sample.flac')
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

import zlib
from os.path import join, basename
from glob import glob
from time import time
from tqdm import tqdm
from cached_property import cached_property
import operator

import numpy as np
import torch
import torch.nn as nn

from ptutils.datasets import get_dataloader, get_dataset_info
from ptutils.utils import RNG, Averager, DictAverager,\
    display, display_level, get_args_decorator, make_hashable, \
    to_numpy


class Trainable(nn.Module):
    """
    Trainable nets should be subclass of this. Implements saving
    checkpoints, controlling random seed etc.
    """
    save_at_times = []
    save_at_epochs = []
    save_dir = "ckpts"

    @get_args_decorator(1)
    def __init__(self, seed, optimiser_type, lr,
                 data_name, save_every_sec=3600,
                 extra_things_to_use_in_hash=tuple(),
                 all_args=None, **nn_kwargs):
        """
        `data_name` is just used to create filename
        """
        args = make_hashable(all_args)
        self.args_hash = str(
            zlib.adler32(
                args.__str__().encode('utf-8')
            ))

        super().__init__()

        # static
        self.init_seed = seed
        self.save_every_sec = save_every_sec
        self.data_name = data_name

        # change throughout training and saved in checkpoints
        self.rng = RNG(seed=seed)
        with self.rng:
            self.init_nn(**nn_kwargs)
            self.optim = optimiser_type(self.parameters(), lr=lr)
        self.epochs = 0
        self.losses = {'train': [], 'valid': []}
        self.logs = {'train': {}, 'valid': {}}
        self.training_time = 0
        # map checkpoints in save_at_times to epochs
        # they were saved at
        self.timed_checkpoint_epochs = {}

        # change throughout training and untracked
        self.last_checkpoint_training_time = 0

    def init_nn(self, **nn_kwargs):

        raise NotImplementedError

    def loss(self, *data):

        # loss should be mean over minibatch
        raise NotImplementedError

    def valid_loss(self, *args, **kwargs):

        return self.loss(*args, **kwargs)

    @property
    def architecture_name(self):

        return str(self.__class__.__name__)

    @property
    def optimiser_name(self):

        return f"{type(self.optim).__name__}_{self.optim.defaults['lr']}"

    @property
    def name(self):

        return f"{self.data_name}_{self.architecture_name}_"+\
            f"{self.optimiser_name}_{self.init_seed}_{self.args_hash}"

    def get_path(self, n_epochs):

        return join(self.save_dir, f"{self.name}_{n_epochs}.checkpoint")

    def get_epochs_from_path(self, path):

        fname = basename(path)
        return int(fname[len(self.name+'_'):-len('.checkpoint')])

    # Can be overridden during loss calculations to store
    # details (with a dictionary of tensors that will be
    # averaged over batches).

    def update_log(self, mode, update):

        if not hasattr(self, 'log'):
            return
        if len(self.logs[mode].keys()) == 0:
            # initialise log
            self.logs[mode] = \
                {k: [] for k in update.keys()}

        # check update and add to log
        assert set(update.keys()) == \
            set(self.logs[mode].keys())
        for key, value in update.items():
            self.logs[mode][key].append(
                to_numpy(value))

    def begin_valid(self):

        self.eval()
        self.valid_rng = RNG(0)
        self.valid_loss_averager = Averager()
        self.valid_log_averager = DictAverager()

    def valid_batch(self, *data, batch_size=None):

        if batch_size is None:
            batch_size = data[0].shape[0]
        with torch.no_grad():
            with self.valid_rng:
                self.valid_loss_averager.include(
                    self.valid_loss(*data).item(), batch_size
                )
                self.valid_log_averager.include(
                    self.log, batch_size
                )

    def end_valid(self):

        self.losses['valid'].append(
            self.valid_loss_averager.avg)
        # average stuff in valid_log
        self.update_log('valid', self.valid_log_averager.avg)

    def begin_epoch(self):

        self.epoch_begin_time = time()
        self.train()

    def uncontrolled_step(self, *data):

        loss = self.loss(*data)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.losses['train'].append(loss.item())
        self.update_log('train', self.log)

    def step(self, *data):

        with self.rng:
            self.uncontrolled_step(*data)

    def end_epoch(self):

        self.epochs += 1
        self.training_time += time() - self.epoch_begin_time
        self.save_if_necessary()

    def save_if_necessary(self):
        """
        Save checkpoint if any of 3 conditions are met:
        - more than a certain time since last save,
        - at a specified number of epochs,
        - first chance to save after a specified time
        """

        time_since_checkpoint = self.training_time\
            - self.last_checkpoint_training_time
        remaining_save_times = [
            save_t for save_t in self.save_at_times
            if save_t > self.last_checkpoint_training_time]
        next_save_at_time = min(remaining_save_times) if\
            len(remaining_save_times) > 0 else np.inf
        is_save_time = self.training_time > next_save_at_time

        if (time_since_checkpoint > self.save_every_sec)\
           or (self.epochs in self.save_at_epochs)\
           or is_save_time:
            self.save_checkpoint()
            self.last_checkpoint_training_time =\
                self.training_time
            if is_save_time:
                self.timed_checkpoint_epochs[next_save_at_time]\
                    = self.epochs

    def save_checkpoint(self):
        display('info', 'Saving Checkpoint.')

        path = self.get_path(self.epochs)
        f = {'optim': self.optim.state_dict(),
             'rng': self.rng.get_state(),
             'params': self.state_dict(),
             'losses': self.losses,
             'epochs': self.epochs,
             'logs': self.logs,
             'training_time': self.training_time,
             'timed_checkpoints': self.timed_checkpoint_epochs}
        torch.save(f, path)

    def load_timed_checkpoint(self, training_time):

        self.load_checkpoint()
        assert training_time in \
            self.timed_checkpoint_epochs, "Time not available."
        self.load_checkpoint(max_epochs=\
            self.timed_checkpoint_epochs[training_time]
        )

    def load_checkpoint(self, max_epochs=None,
                        training_time=None):
        """
        loads most recent checkpoint if max_epochs is None,
        otherwise most recent with less epochs than this
        """
        assert (max_epochs is None)\
            or (training_time is None)

        template = self.get_path(n_epochs='*')
        matches = {self.get_epochs_from_path(path): path
                   for path in glob(template)}
        if max_epochs is not None:
            matches = {n: p for n, p in matches.items()
                       if n <= max_epochs}
        if len(matches) == 0:
            display("info", "No checkpoints found.")
            return
        checkpoint = matches[max(matches)]
        try:
            f = torch.load(checkpoint)
        except EOFError as err:
            print("Failed to open:", checkpoint)
            raise err
        self.rng.set_state(f['rng'])
        self.optim.load_state_dict(f['optim'])
        self.load_state_dict(f['params'])
        self.losses = f['losses']
        self.logs = f['logs']
        self.epochs = f['epochs']
        self.training_time = f['training_time']
        self.timed_checkpoint_epochs = f['timed_checkpoints']
        display("info",
                f"Loaded network trained for {self.epochs}"+\
                f" epochs in {self.training_time} seconds.")
        self.eval()


class ImageClassifier(Trainable):
    """
    Subclass of Trainable designed for image classifiers with standard
    data loaders etc which make training simpler. Controls randomness
    of data loader along with randomness of everything else. Also
    sends data to correct device.
    """

    best_valid_op = min # used to decide if valid loss is new best


    def __init__(self, dataset_name, batch_size,
                 *args, dataloader_kwargs={}, **kwargs):

        for key, value in get_dataset_info(dataset_name).items():
            setattr(self, key, value)
            # img_shape, img_channels, n_classes, loss_weights

        self.short_dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self.dataset_name = f"{dataset_name}_{batch_size}"
        # all arguments are sent to parent to make hash
        # for saving
        super().__init__(
            *args, **kwargs,
            data_name=f"{dataset_name}_{batch_size}",
            extra_things_to_use_in_hash=dataloader_kwargs
        )

    @cached_property
    def train_loader(self):

        return get_dataloader(self.short_dataset_name,
                              self.batch_size, 'train',
                              valid_proportion=0.1,
                              **self.dataloader_kwargs)

    @cached_property
    def valid_loader(self):

        return get_dataloader(self.short_dataset_name,
                              self.batch_size, 'valid',
                              valid_proportion=0.1,
                              **self.dataloader_kwargs)

    def validate(self):

        self.begin_valid()
        # we don't use self.valid_batch as it doesn't
        # manage dataloader's random seed
        with torch.no_grad():
            with RNG(0):
                for data in self.valid_loader:
                    data = self.send_all_to_own_device(*data)
                    batch_size = data[0].shape[0]
                    self.valid_loss_averager.include(
                        self.valid_loss(*data).item(),
                        batch_size
                    )
                    self.valid_log_averager.include(
                        self.log, batch_size
                    )
        self.end_valid()
        display("info", "validation:", self.losses['valid'][-1])

    def got_best_valid(self):

        valids = self.losses["valid"]
        if len(valids) <= 1:
            return False
        return valids[-1] == self.best_valid_op(valids)

    def train_epoch(self):

        self.begin_epoch()
        with self.rng:
            with tqdm(total=len(self.train_loader)) as pbar:
                for data in self.train_loader:
                    data = self.send_all_to_own_device(*data)
                    self.uncontrolled_step(*data)
                    pbar.update(1)
        self.validate()
        self.end_epoch()

    def train_n_epochs(self, max_epochs):

        self.load_checkpoint()
        if self.epochs == 0:
            self.validate()
        while self.epochs < max_epochs:
            self.train_epoch()
            if self.got_best_valid():
                display("info", f"New best validation after {self.epochs} epochs.")
                self.save_checkpoint()
        self.save_checkpoint()

    @property
    def device(self):
        # this assumes all of self is on same device
        return next(self.parameters()).device

    def send_all_to_own_device(self, *args):
        return tuple(arg.to(self.device) for arg in args)

    def load_best_checkpoint(self):
        with display_level(print_info=False):
            self.load_checkpoint()
        valids = self.losses['valid']
        if len(valids) == 0:
            raise Exception("No saved checkpoints.")
            return
        best_n_epochs = valids.index(self.best_valid_op(valids))
        self.load_checkpoint(max_epochs=best_n_epochs)
        assert self.epochs == best_n_epochs, \
            f"No checkpoints saved at best valid loss "+\
            f"(after {best_n_epochs} epochs)."

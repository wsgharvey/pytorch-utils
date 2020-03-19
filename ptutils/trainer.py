from os.path import join, basename
from glob import glob
from time import time
from tqdm import tqdm
from cached_property import cached_property
import operator
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ptutils.datasets import get_dataloader, get_dataset_info
from ptutils.utils import RNG, Averager, DictAverager,\
    display, display_level, get_args_decorator, robust_hash, \
    to_numpy, IntSeq, state_dict_to


class Trainable(nn.Module):
    """
    Trainable nets should be subclass of this. Implements saving
    checkpoints, controlling random seed etc.
    """
    save_dir = "ckpts"

    @get_args_decorator(1)
    def __init__(self, seed, optim_type, optim_kwargs,
                 data_name, extra_things_to_use_in_hash=tuple(),
                 all_args=None, **nn_kwargs):
        """
        `data_name` is just used to create filename
        """
        self.args_hash = robust_hash(all_args)

        super().__init__()

        # static
        self.init_seed = seed
        self.data_name = data_name
        self.init_save_valid_conditions()

        # change throughout training and saved in checkpoints
        self.rng = RNG(seed=self.init_seed)
        with self.rng:
            self.init_nn(**nn_kwargs)
            self.optim = optim_type(self.parameters(), **optim_kwargs)
        self.epochs = 0
        self.losses = {'train': [], 'valid': []}
        self.logs = {'train': {}, 'valid': {}}
        self.training_time = 0
        # structures to track validation/saving by mapping valid/save
        # epochs to list of times they represent
        self.timed_valid_epochs = OrderedDict([])
        self.timed_save_epochs = OrderedDict([])

        # change throughout training and untracked
        None

    def init_save_valid_conditions(self):
        self.save_at_times = IntSeq([])
        self.save_at_epochs = IntSeq([])
        self.valid_at_times = IntSeq([])
        self.valid_at_epochs = IntSeq([])

    def set_save_valid_conditions(self, save_or_valid,
                                  every_or_eachof, N, sec_or_epochs):
        """
        set condition to save/validate wither every so often or at specified epochs/times
        """
        def get_bool(arg, is_this, not_this):
            return {is_this: True, not_this: False}[arg]
        is_save = get_bool(save_or_valid, 'save', 'valid')
        is_every = get_bool(every_or_eachof, 'every', 'eachof')
        is_sec = get_bool(sec_or_epochs, 'sec', 'epochs')

        if is_every:
            def get_seq():
                x = 0
                while True:
                    yield x
                    x += N
            seq = get_seq()
        else:
            seq = N
        add_to = {
            (False, False): self.valid_at_epochs,
            (False, True): self.valid_at_times,
            (True, False): self.save_at_epochs,
            (True, True): self.save_at_times,
        }[(is_save, is_sec)]
        add_to.mix_in_iterable(seq)

    def init_nn(self, **nn_kwargs):

        raise NotImplementedError

    def loss(self, *data):

        # loss should be mean over minibatch
        raise NotImplementedError

    def eval_metric(self, *args, **kwargs):
        # valid / eval metric

        return self.loss(*args, **kwargs)

    @property
    def architecture_name(self):

        return str(self.__class__.__name__)

    @property
    def optimiser_name(self):

        return f"{type(self.optim).__name__}_{robust_hash(self.optim.defaults)}"

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
        self.log = {}
        self.eval_rng = RNG(0)
        self.valid_metric_averager = Averager()
        self.valid_log_averager = DictAverager()

    def uncontrolled_eval_batch(self, *data, batch_size=None):

        if batch_size is None:
            batch_size = data[0].shape[0]
        with torch.no_grad():
            metric = self.eval_metric(*data).item()
            self.valid_metric_averager.include(
                metric, batch_size
            )
            self.valid_log_averager.include(
                self.log, batch_size
            )

    def eval_batch(self, *args, **kwargs):

        with self.eval_rng:
            return self.uncontrolled_eval_batch(*args, **kwargs)

    def begin_eval(self):

        self.testing = True
        return self.begin_valid()

    def end_eval(self):

        assert self.testing
        self.testing = False
        display('info', f'returned: {self.valid_metric_averager.avg}')
        display('info', f'logged: {self.valid_log_averager.avg}')
        return self.valid_metric_averager.avg, self.valid_log_averager.avg

    def end_valid(self):

        self.losses['valid'].append(
            self.valid_metric_averager.avg)
        # average stuff in valid_log
        self.update_log('valid', self.valid_log_averager.avg)

    def begin_epoch(self):

        self.epoch_begin_time = time()
        self.train()
        self.log = {}

    def uncontrolled_step(self, *data):

        start = time()
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
        self.valid_and_save_if_necessary()

    def applicable_action_times(self, at_epochs, at_times,
                                timed_epochs):
        """
        - action is either save or valid
        - if action is required, returns (possibly empty)
          list of applicable times from at_times
        - otherwise, returns None
        """
        all_prev_epochs = timed_epochs.keys()
        if self.epochs in all_prev_epochs:
            return None  # already validated at this point
        all_prev_times = sum(timed_epochs.values(), [0.])
        prev_time = all_prev_times[-1]
        applicable_times = list(at_times.all_between(
            prev_time, self.training_time,
            closed_lower=False, closed_upper=True
        ))
        if len(applicable_times) > 0:
            return applicable_times
        elif self.epochs in at_epochs:
            return []
        else:
            return None

    def action_if_necessary(self, at_epochs, at_times,
                            timed_epochs, action):

        times = self.applicable_action_times(
            at_epochs, at_times, timed_epochs)
        if times is not None:
            timed_epochs[self.epochs] = times
            action()

    def valid_and_save_if_necessary(self):

        # validation
        self.action_if_necessary(
            self.valid_at_epochs, self.valid_at_times,
            self.timed_valid_epochs, self.validate
        )
        # saving
        self.action_if_necessary(
            self.save_at_epochs, self.save_at_times,
            self.timed_save_epochs, self.save_checkpoint
        )

    def save_checkpoint(self):

        display('info', 'Saving Checkpoint.')
        path = self.get_path(self.epochs)
        f = {'optim': self.optim.state_dict(),
             'rng': self.rng.get_state(),
             'params': self.state_dict(),
             'epochs': self.epochs,
             'losses': self.losses,
             'logs': self.logs,
             'training_time': self.training_time,
             'timed_valid_epochs': self.timed_valid_epochs,
             'timed_save_epochs': self.timed_save_epochs,
             }
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
            display('error', f"Failed to open: {checkpoint}")
            raise err
        self.rng.set_state(f['rng'])
        self.optim.load_state_dict(f['optim'])
        self.load_state_dict(f['params'])
        self.epochs = f['epochs']
        self.losses = f['losses']
        self.logs = f['logs']
        self.training_time = f['training_time']
        self.timed_valid_epochs = f['timed_valid_epochs']
        self.timed_save_epochs = f['timed_save_epochs']
        display("info",
                f"Loaded network trained for {self.epochs}"+\
                f" epochs in {self.training_time} seconds.")
        self.eval()


class HasDataloaderMixin():
    """
    Mix-in for trainable objects which are simply trained by feeding
    data from a data-loader into .step(...)
    """

    @staticmethod
    def iter_tupled(iterator):
        """
        - if items in sequence are not tuples, wrap them in tuples
        - needed so that functions returning single item are treated
        similarly to functions returning multiple
        """
        for item in iterator:
            if isinstance(item, tuple) or isinstance(item, list):
                yield item
            else:
                yield (item,)

    def set_dataloaders(self, train_loader, valid_loader, eval_loader):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader

    def train_epoch(self):

        self.begin_epoch()
        with self.rng:
            with tqdm(total=len(self.train_loader)) as pbar:
                for data in self.iter_tupled(self.train_loader):
                    self.uncontrolled_step(*data)
                    pbar.update(1)
        self.end_epoch()

    def train_n_epochs(self, max_epochs):

        self.load_checkpoint()
        self.valid_and_save_if_necessary()
        while self.epochs < max_epochs:
            self.train_epoch()
        self.save_checkpoint()

    def validate(self):

        self.begin_valid()
        with self.eval_rng:
            for data in self.iter_tupled(self.valid_loader):
                self.uncontrolled_eval_batch(*data)
        self.end_valid()
        display("info", "validation:", self.losses['valid'][-1])

    def evaluate(self):

        self.begin_eval()
        with self.eval_rng:
            for data in self.iter_tupled(self.eval_loader):
                self.uncontrolled_eval_batch(*data)
        return self.end_eval()


class CudaCompatibleMixin():
    """
    A mixin for Trainables that can be run on GPU by simply moving all
    parameters to GPU. Controls moving data to GPU
    and saving checkpoints on CPU (for easy loading on any device).
    """
    @property
    def device(self):
        # this assumes all of self is on same device
        return next(self.parameters()).device

    def to_cuda(self):

        self.cuda()
        self.optim_to_device(self.device)

    def to_cpu(self):

        self.cpu()
        self.optim_to_device(self.device)

    def is_cuda(self):

        return self.device.type == 'cuda'

    def send_all_to_own_device(self, args):
        def to_device_if_tensor(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(self.device)
            return arg
        return tuple(map(to_device_if_tensor, args))

    def optim_to_device(self, device):
        self.optim.load_state_dict(
            state_dict_to(
                self.optim.state_dict(),
                self.device))

    def on_cpu_wrapper(f):
        def wrapped_f(self, *args, **kwargs):
            was_cuda = self.is_cuda()
            self.to_cpu()
            result = f(self, *args, **kwargs)
            if was_cuda:
                self.to_cuda()
            return result
        return wrapped_f

    def args_to_device_wrapper(f):
        def wrapped_f(self, *args):
            args = self.send_all_to_own_device(args)
            return f(self, *args)
        return wrapped_f

    # this mixin must be left of Trainable in class definition to make the below work

    @on_cpu_wrapper
    def save_checkpoint(self, *args, **kwargs):
        super().save_checkpoint(*args, **kwargs)

    @on_cpu_wrapper
    def load_checkpoint(self, *args, **kwargs):
        super().load_checkpoint(*args, **kwargs)

    @args_to_device_wrapper
    def uncontrolled_step(self, *data):
        super().uncontrolled_step(*data)

    @args_to_device_wrapper
    def uncontrolled_eval_batch(self, *data):
        super().uncontrolled_eval_batch(*data)

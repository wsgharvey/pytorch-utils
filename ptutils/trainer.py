import os
from glob import glob
from time import time
from tqdm import tqdm
import copy
import functools
from cached_property import cached_property
import operator
from collections import OrderedDict
import wandb

import numpy as np
import torch
import torch.nn as nn

from ptutils.datasets import get_dataloader, get_dataset_info
from ptutils.utils import RNG, Averager, DictAverager,\
    get_args_decorator, robust_hash, to_numpy, IntSeq,\
    state_dict_to, make_hashable
from exputils import display, display_level


class Trainable(nn.Module):
    """
    Trainable nets should be subclass of this. Implements saving
    checkpoints, controlling random seed etc.
    """
    save_dir = "ckpts"

    def add_logger(self, attr, getter=None, setter=None,
                   on_cuda=False):
        # on_cuda:  whether to move to gpu for CUDA training

        if not hasattr(self, 'logged_attrs'):
            self.logged_attrs = {}
            self.cuda_attrs = set()

        if getter is None:
            def getter(obj):
                return getattr(obj, attr)
        if setter is None:
            def setter(obj, value):
                setattr(obj, attr, value)
        self.logged_attrs[attr] = (getter, setter)
        if on_cuda:
            self.cuda_attrs.add(attr)

    @get_args_decorator(1)
    # '*' makes following args keyword args
    def __init__(self, *, seed, nn_args, optim_args,
                 name_prefix='',
                 extra_things_to_use_in_hash=tuple(),
                 all_args):

        self.all_kwargs = all_args[1]
        self.args_hash = robust_hash(all_args)

        super().__init__()

        # static
        self.init_seed = seed
        self.name_prefix = name_prefix
        self.init_save_valid_conditions()

        # things we need to keep track of (and log) throughout training
        self.epochs = 0
        self.add_logger('epochs')
        self.losses = {'train': [], 'valid': []}
        self.add_logger('losses')
        self.logs = {'train': {}, 'valid': {}}
        self.add_logger('logs')
        self.training_time = 0
        self.add_logger('training_time')
        # structures to track validation/saving by mapping valid/save
        # epochs to list of times they represent
        self.timed_valid_epochs = OrderedDict([])
        self.add_logger('timed_valid_epochs')
        self.timed_save_epochs = OrderedDict([])
        self.add_logger('timed_save_epochs')

        # initialise RNG, weights and optimiser
        self.rng = RNG(seed=self.init_seed)
        with self.rng:
            self.init_nn(**nn_args)
            self.init_optim(**optim_args)
        # add them to logs
        self.add_logger(
            'rng', lambda self: self.rng.get_state(),
            lambda self, state: self.rng.set_state(state),)
        self.add_logger(
            'weights', lambda self: self.state_dict(),
            lambda self, state: self.load_state_dict(state),
            on_cuda=True,)
        self.add_logger(
            'optim', lambda self: self.get_optim_state(),
            lambda self, state: self.set_optim_state(state),
            on_cuda=True,)

        self.post_init()

    def get_optim_state(self):
        return self.optim.state_dict()

    def set_optim_state(self, state):
        return self.optim.load_state_dict(state)

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

    def init_optim(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        optim_type = kwargs.pop('type')
        self.optim = optim_type(self.parameters(), **kwargs)

    def post_init(self):

        pass

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

        return f"{self.name_prefix}_{self.architecture_name}_"+\
            f"{self.optimiser_name}_{self.init_seed}_{self.args_hash}"

    def get_path(self, n_epochs):

        return os.path.join(self.save_dir, f"{self.name}_{n_epochs}.checkpoint")

    @property
    def path(self):

        return self.get_path(self.epochs)

    @ property
    def num_iterations(self):

        return len(self.losses['train'])

    def get_epochs_from_path(self, path):

        fname = os.path.basename(path)
        return int(fname[len(self.name+'_'):-len('.checkpoint')])

    # Can be overridden during loss calculations to store
    # details (with a dictionary of tensors that will be
    # averaged over batches).

    def update_log(self, mode, update):

        # if not hasattr(self, 'log'):
        #     return
        if len(self.logs[mode].keys()) == 0:
            # initialise log
            self.logs[mode] = \
                {k: [] for k in update.keys()}

        # check update and add to log
        assert set(update.keys()) == \
            set(self.logs[mode].keys())
        for key, value in update.items():
            self.logs[mode][key].append(to_numpy(value))

    def begin_valid_eval(self):

        self.eval()
        self.log = {}
        self.eval_rng = RNG(0)
        self.valid_metric_averager = Averager()
        self.valid_log_averager = DictAverager()

    def begin_valid(self):

        return self.begin_valid_eval()

    def begin_eval(self):

        self.testing = True    # TODO change name of self.testing?
        return self.begin_valid_eval()

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

    def end_valid_eval(self):

        pass

    def log_extra_validation(self):
        """
        Method can be overwritten for logging things that
        shouldn't be averaged over validation batches.
        """
        pass

    def end_valid(self):

        self.end_valid_eval()
        self.losses['valid'].append(
            self.valid_metric_averager.avg)
        # average stuff in valid_log
        self.log = self.valid_log_averager.avg
        self.log_extra_validation()
        self.update_log('valid', self.log)

    def end_eval(self):

        self.end_valid_eval()
        assert self.testing
        self.testing = False
        display('info', f'returned: {self.valid_metric_averager.avg}')
        display('info', f'logged: {self.valid_log_averager.avg}')
        return self.valid_metric_averager.avg, self.valid_log_averager.avg

    def begin_epoch(self):

        self.epoch_begin_time = time()
        self.train()
        self.log = {}

    def uncontrolled_step(self, *data):

        start = time()
        loss = self.loss(*data)
        self.optim.zero_grad()
        loss.backward()
        self.modify_gradients()
        self.optim.step()
        self.losses['train'].append(loss.item())
        self.update_log('train', self.log)

    def modify_gradients(self):
        pass

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


    def get_trainable_state(self):

        ckpt = {}
        for attr, (getter, setter) in  self.logged_attrs.items():
            ckpt[attr] = getter(self)
        return ckpt

    def set_trainable_state(self, ckpt, allow_missing_attrs=False):

        for attr, (getter, setter) in self.logged_attrs.items():
            if attr in ckpt:
                setter(self, ckpt[attr])
            elif not allow_missing_attrs:
                raise Exception(f'Missing {attr} from state dict.')

    def save_checkpoint(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        display('info', 'Saving Checkpoint.')
        torch.save(
            self.get_trainable_state(),
            self.path,
        )

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
            t = torch.load(checkpoint)
        except EOFError as err:
            display('error', f"Failed to open: {checkpoint}")
            raise err
        self.set_trainable_state(t)
        display("info",
                f"Loaded network trained for {self.epochs}"+\
                f" epochs in {self.training_time} seconds.")
        self.eval()

    def load_best_checkpoint(self):
        with display_level(print_info=False):
            self.load_checkpoint()
        valids = self.losses['valid']
        if len(valids) == 0:
            raise Exception("No saved checkpoints.")
            return
        assert hasattr(self, 'best_evaluation_op'), \
            "Must specify best_evaluation_op (probably `max` or `min`)."
        best_index = valids.index(self.best_evaluation_op(valids))
        best_n_epochs = list(self.timed_valid_epochs.keys())[best_index]
        self.load_checkpoint(max_epochs=best_n_epochs)
        assert self.epochs == best_n_epochs, \
            f"No checkpoints saved at best valid loss "+\
            f"(after {best_n_epochs} epochs)."

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
                    self.tqdm_text = None
                    self.uncontrolled_step(*data)
                    pbar.update(1)
                    if self.tqdm_text is not None:
                        pbar.set_description(self.tqdm_text)
        self.end_epoch()

    def train_n_epochs(self, max_epochs):

        self.load_checkpoint(max_epochs=max_epochs)
        self.valid_and_save_if_necessary()
        start_epochs = self.epochs
        while self.epochs < max_epochs:
            self.train_epoch()
        if self.epochs > start_epochs:
            self.save_checkpoint()

    def uncontrolled_full_eval(self, loader):

        for data in self.iter_tupled(loader):
            self.uncontrolled_eval_batch(*data)

    def validate(self):

        self.begin_valid()
        with self.eval_rng:
            self.uncontrolled_full_eval(self.valid_loader)
        self.end_valid()
        display("info", "validation:", self.losses['valid'][-1])

    def evaluate(self):

        self.begin_eval()
        with self.eval_rng:
            self.uncontrolled_full_eval(self.eval_loader)
        return self.end_eval()


class CudaCompatibleMixin():
    """
    A mixin for Trainables that can be run on GPU by simply moving all
    parameters to GPU. Controls moving data to GPU
    and saving checkpoints on CPU (for easy loading on any device).
    This mixin should be given to left of Trainable in class definition.
    """
    @property
    def device(self):
        return next(self.parameters()).device

    def move_all_to_own_device(self):

        state = self.get_trainable_state()
        cuda_state = {k: v for k, v in state.items()
                      if k in self.cuda_attrs}
        self.set_trainable_state(
            state_dict_to(
                cuda_state,
                self.device
            ),
            allow_missing_attrs=True
        )

    def to_cuda(self):

        self.cuda()
        self.move_all_to_own_device()

    def to_cpu(self):

        self.cpu()
        self.move_all_to_own_device()

    def is_cuda(self):

        return self.device.type == 'cuda'

    def send_args_to_own_device(self, args):

        def to_device_if_tensor(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(self.device)
            return arg
        return tuple(map(to_device_if_tensor, args))

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
            args = self.send_args_to_own_device(args)
            return f(self, *args)
        return wrapped_f

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


class WandbMixin():
    """
    A mixin to log progress with wandb. If a run is restarted, the original run and
    restart should automatically be grouped together on wandb.

    This sends attributes in self.log to wandb rather than storing them locally. self.losses
    are stored both locally and at wandb.
    """
    log_gradient_freq = None  # time period between logging (wandb calls it freq.)
    wandb_init_kwargs = {}

    def post_init(self):

        self.init_wandb(self.wandb_project,)
        if self.log_gradient_freq is not None:
            wandb.watch(self, log='all', log_freq=self.log_gradient_freq)

    def init_wandb(self, wandb_project):

        wandb.init(
            project=wandb_project,
            group=self.wandb_group_name,
            reinit=True,  # TODO what if we want to run a script with several
                          # Trainables? does it work?,
            config=make_hashable(self.all_kwargs, make_dict=True),
            **self.wandb_init_kwargs,
        )

    @property
    def wandb_group_name(self):

        return self.name

    def update_log(self, mode, metrics):
        """
        syncs to wandb instead of storing log
        """
        training_time = self.training_time
        if mode == 'train':
            training_time += time() - self.epoch_begin_time
        wandb_log = {
            'time': training_time,
            'epoch': self.epochs,
            'iter': self.num_iterations,  # should be same as wandb's step
        }
        if mode == 'train':
            wandb_log['iter'] = wandb_log['iter'] - 1
        wandb_log.update(
            {f'{mode}-{k}': v if type(v) in [wandb.Image, wandb.Video] else to_numpy(v)
             for k, v in metrics.items()}
        )
        wandb_log[f'{mode}-loss'] = self.losses[mode][-1]

        increment_step = (mode == 'train')
        wandb.log(
            wandb_log,
            commit=increment_step,
        )

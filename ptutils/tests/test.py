import os
import shutil
from functools import wraps
from numbers import Number
import copy

import torch
import torch.nn as nn

from ptutils import Trainable, HasDataloaderMixin, CudaCompatibleMixin

from unittest import TestCase


test_save_dir = 'test_ckpts'
def clean_save_dir_wrapper(foo):
    @wraps(foo)
    def wrapped(*args, **kwargs):
        # set up save directory
        if os.path.exists(test_save_dir):
            shutil.rmtree(test_save_dir)
        os.mkdir(test_save_dir)

        result = foo(*args, **kwargs)

        # delete save directory
        shutil.rmtree(test_save_dir)

        return result
    return wrapped

# basic tests ----------------------------------------

class TestSubclass(Trainable):
    save_dir = test_save_dir
    default_init_kwargs = dict(
        seed=123,
        name_prefix='unit-test',
        optim_args=dict(type=torch.optim.Adam, lr=3e-4),
        nn_args=dict(h=64),
    )

    @staticmethod
    def easy_init():
        return TestSubclass(
            **TestSubclass.default_init_kwargs
        )

    def init_nn(self, h):

        self.net = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def loss(self, x):

        loss = self.net(x).mean()
        self.log = {'loss': loss,
                    'two': torch.tensor(2.)}
        return loss

    def eval_metric(self, *args, **kwargs):

        m = super().eval_metric(*args, **kwargs)
        self.log['is_valid'] = 1
        return m

    def validate(self):

        self.begin_valid()
        for _ in range(2):
            data = torch.randn(8, 1)
            self.eval_batch(data)
        self.end_valid()

    def train_one_epoch(self):

        self.begin_epoch()
        nb, bs = 5, 8
        data = torch.arange(nb*bs)\
                    .view(nb, bs, 1)\
                    .float()
        for batch in data:
            self.step(batch)
        self.end_epoch()


class Tester(TestCase):

    def test_init(self):

        tr = TestSubclass.easy_init()

    def test_train(self):

        tr = TestSubclass.easy_init()
        for _ in range(3):
            tr.train_one_epoch()
        self.assertTrue(tr.epochs == 3)

    def test_validation(self):

        tr = TestSubclass.easy_init()
        tr.set_save_valid_conditions(
            'valid', 'every', 2, 'epochs')
        tr.set_save_valid_conditions(
            'valid', 'every', 3, 'epochs')
        tr.set_save_valid_conditions(
            'valid', 'eachof', [4, 5, 6], 'epochs'
        )
        for _ in range(9):
            tr.train_one_epoch()
        save_epochs = list(tr.timed_valid_epochs.keys())
        self.assertTrue(save_epochs == [2, 3, 4, 5, 6, 8, 9])

        # check validation logs
        valid_log = tr.logs['valid']
        self.assertTrue(len(valid_log['two']) == 7)
        self.assertTrue(all(item == 2 for item in valid_log['two']))

    @clean_save_dir_wrapper
    def test_saving(self):

        tr = TestSubclass.easy_init()
        tr.set_save_valid_conditions(
            'save', 'every', 2, 'epochs')
        for _ in range(10):
            tr.train_one_epoch()
        files = os.listdir(tr.save_dir)
        self.assertTrue(len(files) == 5)


# test HasDataloaderMixin --------------------------

class TestDataloaderNet(HasDataloaderMixin, TestSubclass):

    # class Loader():

    #     def __init__(self, data):

    #         self.data = data

    #     def __len__(self):

    #         return len(self.data)

    #     def __iter__(self):

    #         for row in self.data:
    #             yield row

    def set_default_dataloaders(self):

        train_loader = torch.randn(5, 1)
        valid_loader = torch.randn(5, 1)
        test_loader = torch.randn(5, 1)
        self.set_dataloaders(
            train_loader, valid_loader, test_loader
        )

    @staticmethod
    def easy_init():

        tr = TestDataloaderNet(
            **TestDataloaderNet.default_init_kwargs
        )
        tr.set_default_dataloaders()
        return tr

class DataloaderTester(TestCase):

    @clean_save_dir_wrapper
    def test_train(self):

        tr = TestDataloaderNet.easy_init()
        tr.train_n_epochs(3)
        self.assertTrue(tr.epochs == 3)

    def test_evaluate(self):

        tr = TestDataloaderNet.easy_init()
        metric, log = tr.evaluate()
        self.assertTrue(isinstance(metric, Number))

# test CudaCompatibleMixin --------------------------

class TestCudaNet(CudaCompatibleMixin, TestSubclass):

    @staticmethod
    def easy_init():
        return TestCudaNet(
            **TestCudaNet.default_init_kwargs
        )


class CudaTester(TestCase):

    def test_train(self):

        tr = TestCudaNet.easy_init()
        tr.cuda()
        for _ in range(3):
            tr.train_one_epoch()
        self.assertTrue(tr.epochs == 3)

    @clean_save_dir_wrapper
    def test_mixed_training(self):
        """
        Check that we can train on mix of CPU and GPU,
        and that we can save on GPU and then load straight
        onto CPU.
        """
        # initialise and train on cpu
        tr = TestCudaNet.easy_init()
        self.assertTrue(tr.device.type == 'cpu')
        tr.train_one_epoch()

        # train on mixture of cpu and gpu
        for send_func, device in [(tr.to_cuda, 'cuda'),
                                  (tr.to_cpu, 'cpu'),
                                  (tr.to_cuda, 'cuda')]:
            send_func()
            self.assertTrue(tr.device.type == device)
            tr.train_one_epoch()

        # save checkpoint from gpu
        tr.save_checkpoint()

        # initialise new model and reload checkpoint
        tr = TestCudaNet.easy_init()
        tr.load_checkpoint()
        tr.train_one_epoch()
        self.assertTrue(tr.device.type == 'cpu')
        self.assertTrue(tr.epochs == 5)

# test deriving class with scheduler LR -------------

class ScheduledNet(TestDataloaderNet):

    def init_optim(self, **kwargs):

        self.optim = torch.optim.Adam(self.parameters(), lr=kwargs['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=kwargs['gamma'])

    def end_epoch(self):
        self.lr_scheduler.step()
        super().end_epoch()

    @staticmethod
    def easy_init():
        init_kwargs = copy.deepcopy(ScheduledNet.default_init_kwargs,)
        init_kwargs['optim_args']['lr'] = 0.1
        init_kwargs['optim_args']['gamma'] = 0.1
        sn = ScheduledNet(
            **init_kwargs
        )
        sn.set_default_dataloaders()
        return sn

    def post_init(self):

        self.add_logger(
            'optim',
            lambda self: {'optim': self.optim.state_dict(),
                          'lr': self.lr_scheduler.state_dict()},
            lambda self, state: (self.optim.load_state_dict(state['optim']),
                                 self.lr_scheduler.load_state_dict(state['lr']),),
            on_cuda=True,)


class SchedulerTester(TestCase):

    @clean_save_dir_wrapper
    def test_train(self):

        tr = ScheduledNet.easy_init()
        def check_lr(n_epochs):
            tr.train_n_epochs(n_epochs)
            lr = tr.optim.state_dict()['param_groups'][0]['lr']
            should_be = 0.1 * 0.1**n_epochs
            approx_eq = lambda a, b: abs(a-b)/(a+b) < 1e-8
            self.assertTrue(approx_eq(lr, should_be))
        check_lr(2)
        check_lr(4)

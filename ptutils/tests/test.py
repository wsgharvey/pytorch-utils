import os
import shutil
from functools import wraps

import torch
import torch.nn as nn

from ptutils import Trainable, CudaCompatibleMixin

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
        seed=1,
        optim_type=torch.optim.Adam,
        optim_kwargs={'lr': 3e-4},
        data_name='unittest',
        h=64)

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

    def valid_metric(self, *args, **kwargs):

        m = super().valid_metric(*args, **kwargs)
        self.log['is_valid'] = 1
        return m

    def validate(self):

        self.begin_valid()
        for _ in range(2):
            data = torch.randn(8, 1)
            self.valid_batch(data)
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
            'save', 'every', 2, 'epochs'
        )
        for _ in range(10):
            tr.train_one_epoch()
        files = os.listdir(tr.save_dir)
        self.assertTrue(len(files) == 5)


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
        print(tr.device.type)
        tr.train_one_epoch()
        self.assertTrue(tr.device.type == 'cpu')
        self.assertTrue(tr.epochs == 5)

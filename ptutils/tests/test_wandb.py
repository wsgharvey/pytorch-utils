import os
from unittest import TestCase
import wandb

from ptutils import WandbMixin
from .test import TestCudaNet, clean_save_dir_wrapper

os.environ['WANDB_SILENT'] = 'true'


class WandbLoggedNet(WandbMixin, TestCudaNet):

    wandb_project = 'test'
    wandb_init_kwargs = dict(dir='test_ckpts')
    log_gradient_freq = 2

    @staticmethod
    def easy_init():
        return WandbLoggedNet(
            **WandbLoggedNet.default_init_kwargs,
        )

    @staticmethod
    def easy_init_other():
        kwargs = WandbLoggedNet.default_init_kwargs
        kwargs['seed'] = kwargs['seed'] + 1
        return WandbLoggedNet(**kwargs)


class WandbTester(TestCase):

    @clean_save_dir_wrapper
    def test_logging(self):

        net = WandbLoggedNet.easy_init()
        net.set_save_valid_conditions('valid', 'every', 1, 'epochs')

        net.train_one_epoch()
        net.train_one_epoch()

        # ensure that we are not saving log in net (should be online instead)
        self.assertTrue(net.logs['train'] == {})
        self.assertTrue(net.logs['valid'] == {})

        net.to_cuda()
        net.train_one_epoch()

        self.assertTrue(wandb.run.project_name() == net.wandb_project)

        url = wandb.run.get_url()

        wandb.join()

        api = wandb.Api()
        path = '/'.join(url.split('/')[-3:])
        run = api.run(path)

        self.assertTrue(run.state == 'finished')
        for key in ['seed', 'nn_args', 'name_prefix']:
            self.assertTrue(run.config[key] == net.default_init_kwargs[key])
        history = run.history(pandas=False)
        self.assertTrue(history[0]['train-two'] == 2)
        self.assertTrue(history[2]['iter'] == history[2]['_step'])
        self.assertTrue(history[-1]['epoch'] == net.epochs)

    @clean_save_dir_wrapper
    def multiple_trainables(self):

        net1 = WandbLoggedNet.easy_init()
        net2 = WandbLoggedNet.easy_init_other()

        # TODO check things are logged correctly for each
        net1.train_one_epoch()
        net1.train_one_epoch()
        net2.train_one_epoch()
        net1.train_one_epoch()


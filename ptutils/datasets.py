import torch
import torch.utils.data as data
import torchvision.datasets as ds
import torchvision.transforms as T

from ptutils.synthetic_datasets import Clocks, DigitalClocks,\
    ShiftedMNIST
from ptutils.street_signs import STS
from ptutils.utils import RNG


def get_dataset_info(name):
    info = {
        'MNIST': {'n_classes': 10,
                  'img_shape': (28, 28,),
                  'img_channels': 1,
                  'loss_weights': None},
        'ShiftedMNIST': {'n_classes': 10,
                         'img_shape': ShiftedMNIST.img_shape,
                         'img_channels': 1,
                         'loss_weights': None},
        'CIFAR10': {'n_classes': 10,
                    'img_shape': (32, 32,),
                    'img_channels': 3,
                    'loss_weights': None},
        'Clocks': {'n_classes': Clocks.n_classes,
                   'img_shape': Clocks.img_shape,
                   'img_channels': 3,
                   'loss_weights': None},
        'DigitalClocks': {'n_classes': DigitalClocks.n_classes,
                          'img_shape': DigitalClocks.img_shape,
                          'img_channels': 3,
                          'loss_weights': None},
        'STS': {'n_classes': STS.n_classes,
                'img_shape': STS.img_shape,
                'img_channels': 3,
                'loss_weights': torch.tensor(
                    [0.3842, 2.3639, 1.5183, 3.1653])}
    }
    return info[name]


normalisations = {
    'MNIST': ((0.1307,), (0.3081,)),
    'CIFAR10': ((0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261)),
    'ShiftedMNIST': ((0.1307,), (0.3081,)),
    'Clocks': ((0.9292, 0.9292, 0.9292),
               (0.2321, 0.2321, 0.2322)),
    'DigitalClocks': ((0.8088, 0.8088, 0.8088),
                      (0.3914, 0.3914, 0.3914)),
    'STS': ((0.4763, 0.5949, 0.6528),
            (0.3300, 0.3595, 0.3784)),
}

def get_dataloader(name, batch_size, mode,
                   valid_proportion,
                   num_workers=4,
                   semisupervised=False):

    datasets = {
        'MNIST': ds.MNIST,
        'ShiftedMNIST': ShiftedMNIST,
        'CIFAR10': ds.CIFAR10,
        'Clocks': Clocks,
        'DigitalClocks': DigitalClocks,
        'STS': STS
    }
    augmentations = {
        'MNIST': [],
        'ShiftedMNIST': [],
        'CIFAR10': [T.RandomHorizontalFlip()],
        'Clocks': [],
        'DigitalClocks': [],
        'STS': [T.RandomApply(     # params from Katharopoulos
            [T.ColorJitter(brightness=0.2),
             T.RandomAffine(0, translate=(100/1280, 100/960))],
            p=0.8)],
    }

    assert name in datasets
    assert mode in ['train', 'valid', 'test']

    transform = (augmentations[name] if
                 mode == 'train' else []) +\
        [T.ToTensor(), T.Normalize(*normalisations[name])]

    dataset = datasets[name](
        root='datasets',
        train=(mode != 'test'),
        download=True,
        transform=T.Compose(transform),
    )
    if semisupervised:
        dataset.set_semisupervision()

    if mode != 'test':
        # do train/valid split

        if hasattr(dataset, 'train_valid_split'):
            train, valid = dataset.train_valid_split(
                valid_proportion)

        else:
            with RNG(0):
                N = len(dataset)
                N_valid = int(N*valid_proportion)
                N_train = N - N_valid
                train, valid = data.random_split(
                    dataset, [N_train, N_valid]
                )

        datasets = {'train': train,
                    'valid': valid}
        dataset = datasets[mode]

    dataloader = data.DataLoader(
        dataset, batch_size,
        shuffle=(mode=='train'),
        drop_last=(mode=='train'),
        pin_memory=True,
        num_workers=num_workers
    )
    return dataloader

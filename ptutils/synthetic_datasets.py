import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets as ds

from ptutils.utils import RNG

def fig2array(fig, h, w):
    fig.canvas.draw()
    flat_array = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype=np.uint8)
    plt.close(fig)
    width, height = fig.canvas.get_width_height()
    array = flat_array.reshape((height, width, 3))
    # now resize with PIL
    img = Image.fromarray(array)\
               .resize((w, h))
    return np.array(img)

class CustomImageDataset(data.Dataset):
    # should define train_length, test_length, n_classes, img_dim

    def __init__(self, root, train,
                 transform, download):

        self.root = root
        self.train = train
        self.transform = transform
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        if not os.path.exists(self.image_path):
            self.make_data()
        self.load_data()

    @property
    def directory(self):
        return os.path.join(
            self.root, self.__class__.__name__
        )

    @property
    def image_path(self):
        fname = 'image_train' if self.train \
            else 'image_test'
        return os.path.join(
            self.directory, fname+'.npy')

    @property
    def label_path(self):
        fname = 'label_train' if self.train \
            else 'label_test'
        return os.path.join(
            self.directory, fname+'.npy')

    def __len__(self):
        return len(self.arrays)

    def load_data(self):

        self.arrays = np.load(self.image_path)
        self.labels = np.load(self.label_path)

    def __getitem__(self, i):
        img = Image.fromarray(self.arrays[i])
        label = torch.tensor(self.labels[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def compute_normalisation(self):
        tensor = torch.stack(
            [self[i][0] for i in range(len(self))],
            dim=0
        )
        return tensor.mean(dim=(0, 2, 3)),\
            tensor.std(dim=(0, 2, 3))

    def make_data(self):

        if self.train:
            nums = range(self.train_length)
        else:
            nums = range(self.train_length,
                         self.train_length+self.test_length)
        arrays, labels = [], []
        for i in nums:
            with RNG(i):
                lab, arr = self.make_pair(i)
            arrays.append(arr)
            labels.append(lab)
        arrays = np.stack(arrays)
        labels = np.stack(labels)
        np.save(self.image_path, arrays)
        np.save(self.label_path, labels)

    def make_pair(self, i):
        # generate a data pair (integer target and numpy array)
        # will be called with random seed set to i
        raise NotImplementedError


class DigitalClocks(CustomImageDataset):
    img_shape = (50, 100)
    n_classes = 24 * 6
    train_length = 20000
    test_length = 2000

    # top middle bottom, top left, top right, bottom left, bottom right
    line_starts = [(2, 0), (1, 0), (0, 0),
                   (1, 0), (1, 1), (0, 0), (0, 1)]
    line_ends = [(2, 1), (1, 1), (0, 1),
                 (2, 0), (2, 1), (1, 0), (1, 1)]
    lines_on = [[1, 0, 1, 1, 1, 1, 1,],
                [0, 0, 0, 0, 1, 0, 1,],
                [1, 1, 1, 0, 1, 1, 0,],
                [1, 1, 1, 0, 1, 0, 1,],
                [0, 1, 0, 1, 1, 0, 1,],
                [1, 1, 1, 1, 0, 0, 1,],
                [1, 1, 1, 1, 0, 1, 1,],
                [1, 0, 0, 0, 1, 0, 1,],
                [1, 1, 1, 1, 1, 1, 1,],
                [1, 1, 0, 1, 1, 0, 1,]]

    def sample_time(self):
        return {'hour': np.random.randint(24),
                'minute': np.random.randint(60),
                'second': np.random.randint(60)}

    def render_time(self, time):

        fig, ax = plt.subplots(figsize=(2, 1))
        ax.set_xlim(0, 7.2)
        ax.set_ylim(-0.2, 2.2)
        ax.axis('off')

        def render_digit(digit, start_x, start_y):

            for on, (dy0, dx0), (dy1, dx1) in zip(
                    self.lines_on[digit],
                    self.line_starts, self.line_ends
            ):
                if on:
                    X = [start_x+dx0, start_x+dx1]
                    Y = [start_y+dy0, start_y+dy1]
                    ax.plot(X, Y, lw=5, color='k')

        digits = [time['hour']//10, time['hour']%10,
                  time['minute']//10, time['minute']%10]
        starts = [0.2, 1.8, 4.0, 5.6]

        for digit, start in zip(digits, starts):

            render_digit(digit, start, 0)

        return fig2array(fig, *self.img_shape)

    def make_pair(self, i):

        time = self.sample_time()
        img = self.render_time(time)
        # target is time in ten minute intervals
        target = 6*time['hour'] + time['minute'] // 10
        return target, img


class Clocks(CustomImageDataset):
    img_shape = (100, 100)
    train_length = 30000
    test_length = 2000

    class Time():
        n_times = 12*60
        round_to = 5

        def __init__(self, i=None):
            if i is None:
                i = np.random.randint(
                    self.n_times)
            else:
                assert i < self.n_times
            self.time = i

        @property
        def minute(self):
            return self.time%60

        @property
        def hour(self):
            return self.time//60

        def add_minutes(self, n):
            self.time += n

        def round(self):
            # round to nearest _ minutes
            rounded = self.round_to *\
                round(self.time/self.round_to)
            return type(self)(rounded%self.n_times)

        def get_target(self):
            return self.round().time//self.round_to

    n_classes = Time.n_times // Time.round_to

    def init_ax(self, ax):
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    def draw_clock(self, ax, time, **kwargs):

        hour_hand_len = 0.4
        minute_hand_len = 0.8
        circle_rad = 1
        minute_ratio = time.minute/60
        hour_ratio = (time.hour+minute_ratio)/12
        ax.add_artist(
            plt.Circle(
                [0, 0], radius=circle_rad,
                fill=False, color='k', lw=2
            )
        )
        colour = 'k' # tuple(np.random.rand(3))
        for length, ratio in zip(
                [minute_hand_len, hour_hand_len],
                [minute_ratio, hour_ratio]):

            end_x = length*np.sin(2*np.pi*ratio)
            end_y = length*np.cos(2*np.pi*ratio)
            ax.plot([0, end_x],
                    [0, end_y],
                    color=colour,
                    lw=3, **kwargs)


    def plot_posterior(self, ax, probs):

        self.init_ax(ax)
        for t, prob in enumerate(probs):
            if prob > 1/2000:
                time = self.Time(
                    t*self.Time.round_to)
                self.draw_clock(
                    ax, time, alpha=prob.item())

    def make_pair(self, i):

        fig, ax = plt.subplots(figsize=(1,1))
        self.init_ax(ax)
        time = self.Time()
        self.draw_clock(ax, time)
        return time.get_target(), \
            fig2array(fig, *self.img_shape)


class ShiftedMNIST(ds.MNIST):
    img_shape = (60, 60)
    digit_dim = 28

    @property
    def raw_folder(self):

        # override to share paths with MNIST
        return os.path.join(
            self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):

        # override to share paths with MNIST
        return os.path.join(
            self.root, 'MNIST', 'processed')

    def __init__(self, train, *args, **kwargs):

        super().__init__(*args, **kwargs)
        seed = 1 if self.train else 2
        with RNG(seed):
            R, C = self.img_shape
            self.coords = np.random.randint(
                (R-self.digit_dim, C-self.digit_dim),
                size=(len(self), 2),
            )

    def __getitem__(self, i):

        img = np.zeros(self.img_shape)
        sub_img = self.data[i].numpy()
        r, c = self.coords[i]
        img[r:r+self.digit_dim,
            c:c+self.digit_dim] = sub_img
        img = Image.fromarray(img)
        target = self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

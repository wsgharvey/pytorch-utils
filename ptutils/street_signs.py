import os
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data

from ptutils.utils import rng_decorator

# based on https://github.com/idiap/attention-sampling/blob/master/scripts/speed_limits.py

class Sign():

    VISIBILITIES = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]

    def __init__(self, string):

        def _float(s):
            try:
                return float(s)
            except ValueError:
                return _float(s[:-1])

        parts = string.split(',')
        self.visibility = self.VISIBILITIES.index(parts[0])
        self.bbox = list(map(_float, parts[1:5]))
        self.name = parts[-1]

    @property
    def area(self):
        return (self.bbox[0]-self.bbox[2])\
            *(self.bbox[1] - self.bbox[3])

    @property
    def visible(self):
        return self.visibility == 0

    def __lt__(self, other):

        # for sorting by visibility - less is more visible
        if self.visibility != other.visibility:
            return self.visibility < other.visibility
        return self.area > other.area

class STS(data.Dataset):

    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]
    n_classes = len(CLASSES)
    img_shape = (960, 1280)

    def _unpackage_if_necessary(self):

        has_zips = any('.zip' in fname for fname
                       in os.listdir(self.directory))
        if has_zips:
            self._unpackage()

    def _unpackage(self):
        # https://www.cvl.isy.liu.se/research/datasets/traffic-signs-dataset/download/
        # if zipped data is downloaded from site above and placed in '{root}/sts/set1'
        # and '{root}/sts/set2' as appropriate, this unzips it.
        files = [
            os.path.join(
                self.directory,
                f'Set{self.set_num}Part{i}.zip')
            for i in [0, 1, 2, 3, 4]
        ]
        for f in files:
            os.system(f'unzip {f} -d {self.directory}')
            os.system(f'rm {f}')

    def __init__(self, root, train, transform, *args, **kwargs):

        self.root = root
        self.train = train
        self.transform = transform
        self.set_num = 1 if self.train else 2
        self.semisupervised = False
        self._unpackage_if_necessary()
        self._preprocess()

    def read_annotation(self, annotation):
        img_path, signs = annotation.split(':')
        signs = signs.replace(' ', '')
        signs = [Sign(string) for string
                 in signs.split(';')
                 if string not in ['', 'MISC_SIGNS']]

        if len(signs) == 0:
            target = 'EMPTY'
            acceptable = True
        else:
            signs = sorted(
                s for s in signs
                if (s.name in self.LIMITS and s.visible)
            )
            if len(signs) == 0:
                target = None
                acceptable = False
            else:
                target = signs[0].name
                acceptable = True

        return img_path, target, acceptable

    def _preprocess(self):
        # target is the most visible sign

        annotations_path = os.path.join(
            self.directory, 'annotations.txt'
        )
        annotations = Path(annotations_path)\
            .read_text().strip('\n').split('\n')
        self.img_paths = []
        self.targets = []
        for annotation in annotations:

            img_path, target, acceptable = \
                self.read_annotation(annotation)
            if acceptable:
                self.img_paths.append(img_path)
                self.targets.append(self.CLASSES.index(target))

        unsupervised = [fname for fname in os.listdir(self.directory)
                        if ('.jpg' in fname and
                            fname not in self.targets)]
        self.img_paths += unsupervised
        self.targets = torch.tensor(self.targets).long()

    def n_with_target(self, target):
        return len([t for t in self.targets.values()
                    if t==target])

    def n_labelled(self):
        return sum(self.n_with_target(i)
                   for i, _ in enumerate(self.CLASSES))

    @property
    def outer_directory(self):

        return os.path.join(
            self.root, 'sts'
        )

    @property
    def directory(self):

        return os.path.join(
            self.outer_directory, f'set{self.set_num}'
        )

    def set_semisupervision(self):

        if self.train:
            self.semisupervised = True

    def __len__(self):

        if self.semisupervised:
            return len(self.img_paths)
        else:
            return len(self.targets)

    def __getitem__(self, i):

        image = self.transform(
            Image.open(
                os.path.join(
                    self.directory, self.img_paths[i]
                )
            )
        )

        if i < len(self.targets):
            return image, self.targets[i]
        else:
            assert self.semisupervised
            return image, torch.LongTensor(self.n_classes)

    def compute_normalisation(self):
        tensor = torch.stack(
            [self[i][0] for i in range(len(self))],
            dim=0
        )
        return tensor.mean(dim=(0, 2, 3)),\
            tensor.std(dim=(0, 2, 3))

    def get_class_weights(self):
        # inverse frequencies using training set, like Katharopoulos

        counts = torch.tensor([
            (self.targets == c).sum() for c in range(self.n_classes)
        ]).float()
        frequencies = counts / counts.sum()
        return 1/(self.n_classes*frequencies)

    @rng_decorator(0)
    def train_valid_split(self, valid_proportion):

        n_valid = int(len(self.targets)*valid_proportion)
        if self.semisupervised:

            valid_indices = np.random.choice(
                len(self.targets), size=n_valid,
                replace=False
            )
            train_indices = [i for i in range(len(self))
                             if i not in valid_indices]
            train = Subset(self, train_indices)
            valid = Subset(self, valid_indices)
            return train, valid

        else:

            return data.random_split(
                self, [len(self)-n_valid, n_valid]
            )

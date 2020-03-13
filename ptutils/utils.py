from collections import OrderedDict
import random
import torch
import numpy as np


# RNG --------------------------------------------------

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)

def get_random_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state()
    }

def set_random_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


class RNG():

    def __init__(self, seed=None, state=None):

        self.state = get_random_state()
        with self:
            if seed is not None:
                set_random_seed(seed)
            elif state is not None:
                set_random_state(state)

    def __enter__(self):
        self.external_state = get_random_state()
        set_random_state(self.state)

    def __exit__(self, *args):
        self.state = get_random_state()
        set_random_state(self.external_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

class rng_decorator():

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            with RNG(self.seed):
                return f(*args, **kwargs)

        return wrapped_f

# printing ---------------------------------------------

def set_display(info=True):

    global print_info
    print_info = True

set_display()   # set with default values

def display(level, *args, **kwargs):

    assert level in [
        "info", "error", "user-requested"
    ]
    global print_info
    if level == "info" and not print_info:
        return

    print(*args, **kwargs)


class display_level():

    def __init__(self, print_info):

        self.inside = print_info

    def __enter__(self):

        global print_info
        self.outside = print_info
        print_info = self.inside

    def __exit__(self, *args, **kwargs):

        global print_info
        print_info = self.outside


class display_level_decorator():

    def __init__(self, *args, **kwargs):
        self.display_args = args
        self.display_kwargs = kwargs

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):

            with display_level(*self.display_args,
                               **self.display_kwargs):

                return f(*args, **kwargs)

        return wrapped_f

# other ------------------------------------------------

def predictions_per_class(pred, y):
    """
    A sort of smooth confusion matrix were, for each class we
    return the average predicted probs over all images in that
    class.
    """
    with torch.no_grad():
        N, C = pred.shape
        dists = torch.zeros(C, C)
        for y_, pred_ in zip(y, pred):

            dists[y_] = pred_
        return dists / dists.sum(dim=1, keepdims=True)

def categorical_entropy(probs):
    """
    Batched entropy calculation for categorical distribution.
    Assumes normalised (and non-log) probs of shape B x C.
    """
    entropy = -(probs*probs.log()).sum(dim=1)
    # set nans to 0 (since 0*log(0) = 0 in entropy calculation)
    entropy[entropy != entropy] = 0
    return entropy

class Averager():
    """
    Support for objects with +, *, and / operators
    (and which can be added to zero).
    """
    def __init__(self):
        self.total = 0
        self.N = 0

    def include(self, mean, n=1):
        self.total = self._add(
            self.total,
            self._mult(mean, n))
        self.N += n

    @property
    def avg(self):
        return self._div(self.total, self.N)

    @staticmethod
    def _add(o1, o2):
        return o1 + o2

    @staticmethod
    def _mult(o, n):
        return o * n

    @staticmethod
    def _div(o, n):
        return o / n


class DictAverager(Averager):
    """
    Allow averaging of dicts containing objects
    with +, *, / operators
    """
    @staticmethod
    def _add(o1, o2):
        if o1 == 0:
            return o2
        assert set(o1.keys()) == set(o2.keys())
        return {k: o1[k]+o2[k]
                for k in o1.keys()}

    @staticmethod
    def _mult(o, n):
        return {k: v*n
                for k, v in o.items()}

    @staticmethod
    def _div(o, n):
        return {k: v/n
                for k, v in o.items()}

class get_args_decorator():

    def __init__(self, start):
        self.start = start

    def __call__(self, foo):

        def wrapped(*args, **kwargs):
            return foo(
                *args, **kwargs,
                all_args=(args[self.start:],
                          kwargs))
        return wrapped

def make_hashable(obj):

    # make objects including nested dictionaries hashable
    t = type(obj)
    if t in [dict, OrderedDict]:
        return tuple(
            make_hashable(entry) for entry in obj.items()
        )
    elif t in [tuple, list]:
        return tuple(
            make_hashable(entry) for entry in obj
        )
    elif t in [str, float, int, bool]:
        return obj
    else:
        try:
            return obj.__repr__()
        except TypeError:
            return str(obj)

# pytorch stuff -----------------------------------

def to_tensor(t):

    if isinstance(t, torch.Tensor):
        return t
    else:
        return torch.tensor(t)

def to_numpy(t):

    # convert tensor
    if isinstance(t, torch.Tensor):
        return np.array(t.detach().cpu())
    else:
        return np.array(t)


def state_dict_to(state_dict, device):

    # TODO debug - maybe something recursive

    # in place operation
    for name, state in state_dict.items():
        state_dict[name] = state.to(device)


class no_grad_decorator():

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            with torch.no_grad():
                return f(*args, **kwargs)
        return wrapped_f

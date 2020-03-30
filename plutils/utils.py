import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np


def get_many_colours(N, n_trials=10):
    """
    Returns N colours, attempting to space them as evenly as possible.
    """
    def sample_colour():
        return tuple(np.random.rand() for _ in range(3))
    def distance(c1, c2):
        # return squared rgb distance
        return sum((v1-v2)**2 for v1, v2 in zip(c1, c2))
    first_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                    (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    if N <= len(first_colours):
        return first_colours[:N]
    colours = first_colours
    for n in range(len(first_colours), N):
        possible_colours = [sample_colour()
                           for _ in range(n_trials)]
        min_distances = [min(distance(colour, possible_colour)
                             for colour in colours)
                         for possible_colour in possible_colours]
        best_index = min_distances.index(max(min_distances))
        colours.append(possible_colours[best_index])
    return colours


def make_x(losses, n_epochs,
           exists_at_zero,
           exists_at_final):
    """
    make x coordinates for regularly calculated loss when we
    just know the number of epochs/fixed-size units of some kind.
    """
    N = len(losses) + (not exists_at_zero) + (not exists_at_final)
    x = np.linspace(0, n_epochs, N)
    if not exists_at_zero:
        x = x[1:]
    if not exists_at_final:
        x = x[:-1]
    return x


def smooth_loss(x, loss, smooth_to):

    assert len(loss) == len(x)
    smooth_every = 1 + len(loss)//smooth_to
    def reshape(array):
        array = np.array(array)
        array = array[:len(array)-len(array)%smooth_every]
        return array.reshape(-1, smooth_every)
    loss = reshape(loss)
    x = reshape(x)
    return x[:, 0], loss.mean(axis=1)


def turn_off_axis(axis, keep_labels=True):
    """
    deactivate matplotlib axis but keep labels
    """
    if not keep_labels:
        axis.axis('off')
        return
    axis.xaxis.set_visible(False)
    plt.setp(axis.spines.values(), visible=False)
    axis.tick_params(left=False, labelleft=False)
    axis.patch.set_visible(False)


def set_yscale(ax, cut_percentile=1, extra_proportion=0.2):
    """
    set_yscale with option to e.g.
    """
    all_ys = np.concatenate([
        line.get_ydata() for line in ax.lines
    ], axis=0)
    percentiles = [cut_percentile, 100-cut_percentile]
    tight_ymin, tight_ymax = np.percentile(
        all_ys, percentiles)
    rang = tight_ymax-tight_ymin
    ymin = tight_ymin-rang*extra_proportion
    ymax = tight_ymax+rang*extra_proportion
    ax.set_ylim(ymin, ymax)

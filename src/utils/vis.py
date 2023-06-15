import os
from pathlib import Path

import torch
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from src import custom_types

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
IMG_DIR = os.path.join(ROOT_DIR, "img")


def _init_plt():
    """Initializes matplotlib."""
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


def _finalize_plt(show=True, save=None):
    """
    :param show: defaults to False
    :param save: name, defaults to None
    """
    if save is not None:
        plt.savefig(os.path.join(IMG_DIR, save), bbox_inches="tight")
    elif show:
        plt.show()


def plot_img(
    data: np.ndarray,
    alphas: list[int] = None,
    min_: list[int] = None,
    max_: list[int] = None,
    plot_channels=False,
    show=True,
    save=None,
):
    # FIXME -

    """Plot np.ndarrays.

    If data is one array and plot_channels=True, then each channel is plotted separately.
    If data is one array and plot_channels=False, then each channel is assigned a color.
    Else each array is assigned a different color (channels collapsed).

    :param data: list[np.ndarray]
    :param plot_channels: defaults to False
    :param show: defaults to False
    :param save: name, defaults to None
    """

    _init_plt()

    for i, channel in enumerate(data):
        a = 1 if alphas is None else alphas[i]
        if min_ is not None and max_ is not None:
            plt.gca().imshow(channel, alpha=a, vmin=min_[i], vmax=max_[i])
        else:
            plt.gca().imshow(channel, alpha=a)

        if plot_channels:
            _finalize_plt(show, save)

    _finalize_plt(show, save)


def plot_imgs(data: np.ndarray, alphas=None, show=True, save=None):
    """
    :param data: 2D ndarray with plot_img() data
    :param alphas: alphas for plot_img(), defaults to None
    :param show: defaults to True
    :param save: name, defaults to None
    """
    _init_plt()

    nrows, ncols = data.shape
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(8, 8),
    )

    d_ = np.stack(np.concatenate(data, axis=0), axis=1)
    min_ = np.min(d_, axis=(1, 2, 3))
    max_ = np.max(d_, axis=(1, 2, 3))

    pbar = tqdm(total=nrows * ncols)
    for i in range(nrows):
        for j in range(ncols):
            cell = data[nrows - i - 1, j]
            plt.sca(axs[i, j])
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
            plot_img(cell, alphas=alphas, min_=min_, max_=max_, show=False, save=None)
            pbar.update(1)

    plt.subplots_adjust(wspace=0, hspace=0)

    _finalize_plt(show, save)
    pbar.close()


def plot_groundtruth(
    dataset, origin, cells_east=10, cells_north=10, show=True, save=None
):
    """Plots groundtruth as grid.
    :param dataset: dataset
    :param origin: grid origin
    :param cells_east: defaults to 5
    :param cells_north: defaults to 5
    :param show: defaults to True
    :param save: defaults to None
    """
    with torch.no_grad():
        data = np.ndarray((cells_east, cells_north), dtype=object)
        _, grid_cells_east, _ = dataset.gridX.shape

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                idx = grid_cells_east * (row + origin[1]) + (col + origin[0])
                data[row, col] = np.vstack(
                    (
                        dataset.X[idx, :, 32 : 2 * 32, 32 : 2 * 32],
                        np.full((32, 32), dataset.Y[idx])[np.newaxis, ...],
                    )
                )

        plot_imgs(data, alphas=[1, 0.5], show=show, save=save)


def plot_predictions(
    net, dataset, origin, cells_east=10, cells_north=10, show=True, save=None
):
    """Plot predictions within grid.
    :param net: trained net
    :param dataset: dataset
    :param origin: grid origin
    :param cells_east: defaults to 5
    :param cells_north: defaults to 5
    :param show: defaults to True
    :param save: defaults to None
    """
    with torch.no_grad():
        data = np.ndarray((cells_east, cells_north), dtype=object)
        _, grid_cells_east, _ = dataset.gridX.shape

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                idx = grid_cells_east * (row + origin[1]) + (col + origin[0])
                y_pred = net(dataset.X[idx, :, :, :][np.newaxis, ...])
                data[row, col] = np.vstack(
                    (
                        dataset.X[idx, :, 32 : 2 * 32, 32 : 2 * 32],
                        np.full((32, 32), y_pred)[np.newaxis, ...],
                    )
                )

        plot_imgs(data, alphas=[1, 0.5], show=show, save=save)

"""
PLOT THE TIMESERIES OF FILTER DISTRIBUTIONS
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from gradient_csmc.utils.printing import ctext

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--D", dest="D", type=int, default=10)
parser.add_argument("--N", dest="N", type=int, default=31)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--i", type=int, default=0)

args = parser.parse_args()


def plot_filter_distribution(true_xs, ys, dists, dirpath):
    true_xs_i = true_xs[args.i]     # (T, D)
    ys_i = ys[args.i]               # (T,)
    dists_i = dists[args.i]         # (T, 2)

    T, dx = true_xs_i.shape
    inds_i = np.arange(dx)
    
    dim = min(10, dx)
    fig, ax = plt.subplots(dim, 1, figsize=(15, 5*dim))
    ax = np.atleast_1d(ax)
    for d in range(dim):
        ys_d = ys_i[inds_i == d]
        ts = np.arange(T)[inds_i == d]
        if ys_d.size > 0:
            ax[d].plot(ts, ys_d, marker="x", color="red", label="Obs", alpha=0.3)

        # plot distributions
        mean = dists_i[d, 0]
        std = dists_i[d, 1]
        ax[d].errorbar(
            d,
            mean,
            yerr=std,
            fmt="o",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            markersize=6,
            alpha=0.9,
            label="Filter mean ±1 std",
        )

        ax[d].plot(np.arange(T), true_xs_i[..., d], color="black", label="Truth")
        ax[d].legend()
        ax[d].set_title(f"Dimension {d}")

    plt.savefig(f"{dirpath}/dists_{args.i}.png")
    plt.close()


def load_data(N: int):
    """ Load data for a given number of particles N"""
    experiment_name = "D={},N={},seed={}"
    experiment_name = experiment_name.format(
        args.D, 
        N, 
        args.seed
    )

    dirpath = f"if-results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        exit()

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath


data, dirpath = load_data(args.N)
plot_filter_distribution(data["true_xs"], data["ys"], data["dists"], dirpath)
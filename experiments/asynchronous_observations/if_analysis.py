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
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--plot-sse", dest="plot_sse", action='store_true')
parser.set_defaults(plot_mae=False, dynamic=False)

args = parser.parse_args()


def plot_filter_distribution(true_xs, ys, dists, joint_dists, dirpath):
    true_xs_i = true_xs[args.i]     # (T, D)
    ys_i = ys[args.i]               # (T,)
    dists_i = dists[args.i]         # (T, 2)
    joint_dists_i = joint_dists[args.i]   # (T, D, 2)

    T, dx = true_xs_i.shape
    inds_i = np.arange(dx)
    
    dim = min(100, dx)
    fig, ax = plt.subplots(dim, 1, figsize=(15, 5*dim))
    ax = np.atleast_1d(ax)
    for d in range(dim):

        # plot distributions
        mean = dists_i[d, 0]
        std = dists_i[d, 1]
        ax[d].errorbar(
            d,
            mean,
            yerr=std,
            fmt="o",
            capsize=5,
            capthick=3,
            elinewidth=3,
            markersize=9,
            alpha=0.9,
            label="Filter mean ±1 std",
        )

        joint_dist_d = joint_dists_i[d, d, :]
        joint_mean = joint_dist_d[0]
        joint_std = joint_dist_d[1]
        ax[d].errorbar(
            d,
            joint_mean,
            yerr=joint_std,
            fmt="o",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            markersize=6,
            alpha=0.9,
            label="Joint filter mean ±1 std",
        )

        ax[d].plot(np.arange(T), true_xs_i[..., d], color="black", label="Truth")

        ys_d = ys_i[inds_i == d]
        ts = np.arange(T)[inds_i == d]
        if ys_d.size > 0:
            ax[d].plot(ts, ys_d, marker="x", color="red", label="Obs", alpha=1)

        ax[d].legend()
        ax[d].set_title(f"Dimension {d}")

    plt.tight_layout()
    plt.savefig(f"{dirpath}/dists_{args.i}.png")
    plt.close()

def plot_dists_sse(dirpath):

    Ns = (31, 128, 246,  512, 1024, 2048, 4096, 8192, 16384)
    best_data, _ = load_data(max(Ns))
    best_dists_i = best_data["dists"][args.i]

    dists_sse = np.empty((len(Ns),))
    joint_dists_sse = np.empty((len(Ns),))

    def find_mae(dists, joint_dists):
        dists_i = dists[args.i]         # (T, 2)
        joint_dists_i = joint_dists[args.i]   # (T, D, 2)
        T, D, _ = joint_dists_i.shape  # T == D
        _dists_sse = 0
        _joint_dists_sse = 0
        for d in range(D):
            _dists_sse += (best_dists_i[d, 0] - dists_i[d, 0]) ** 2
            _joint_dists_sse += (best_dists_i[d, 0] - joint_dists_i[d, d, 0]) ** 2
        return _dists_sse, _joint_dists_sse
    
    for x, n in enumerate(Ns):
        data, _ = load_data(n)
        d, j = find_mae(data["dists"], data["joint_dists"])
        dists_sse[x] = d
        joint_dists_sse[x] = j

    print(dists_sse)
    plt.figure(figsize=(15, 5))
    plt.plot(Ns, np.log(dists_sse + 1e-6), label="Dists log SSE")
    plt.plot(Ns, np.log(joint_dists_sse + 1e-6), label="Joint Dists log SSE")
    plt.title("Convergence of particle filters to good estimates")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{dirpath}/sse.png")
    plt.close()


def load_data(N: int):
    """ Load data for a given number of particles N"""
    experiment_name = "D={},N={},dynamic={},seed={}"
    experiment_name = experiment_name.format(
        args.D,
        N,
        args.dynamic,
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
plot_filter_distribution(data["true_xs"], data["ys"], data["dists"], data["joint_dists"], dirpath)
if args.plot_sse:
    plot_dists_sse("if-results/plots")
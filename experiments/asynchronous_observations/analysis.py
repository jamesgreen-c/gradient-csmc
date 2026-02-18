import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.asynchronous_observations.kernels import KernelType
from gradient_csmc.utils.printing import ctext

from jax.scipy.special import logsumexp

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=1_000)
parser.add_argument("--D", dest="D", type=int, default=10)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--i", type=int, default=0)

args = parser.parse_args()

kernel_type = KernelType(args.kernel)

experiment_name = "kernel={},D={},T={},N={},logvar={},style={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.D, 
    args.T, 
    args.N,
    args.log_var,
    args.style, 
    args.seed
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    print(ctext("No such experiment exists", "yellow"))
    exit()

PLOTDIR = f"results/{experiment_name}/plots"
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


def plot_lineage(ancestors):
    A = ancestors[args.i]          # (M, T)
    x = np.arange(A.shape[-1])     # (T,)
    plt.figure(figsize=(15, 5))
    plt.plot(x, A.T, color="blue", alpha=0.1)
    plt.savefig(f"{PLOTDIR}/lineages_{args.i}.png")
    plt.close()


def plot_ess(ess):
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(ess.shape[-1]), ess[args.i].mean(axis=0))
    plt.savefig(f"{PLOTDIR}/ess_{args.i}.png")
    plt.close()


def plot_smoothing_means(true_xs, xs, ys, inds):
    true_xs_i = true_xs[args.i]
    xs_i = xs[args.i]
    ys_i = ys[args.i]
    inds_i = inds[args.i]

    m, T, dx = xs_i.shape

    chain_means = xs_i.mean(axis=0)
    
    dim = min(10, dx)
    fig, ax = plt.subplots(dim, 1, figsize=(15, 5*dim))
    ax = np.atleast_1d(ax)
    for d in range(dim):
        ys_d = ys_i[inds_i == d]
        ts = np.arange(T)[inds_i == d]
        if ys_d.size > 0:
            ax[d].plot(ts, ys_d, marker="x", color="red", label="Obs", alpha=0.3)

        ax[d].plot(np.arange(T), chain_means[..., d], label="Mean")
        ax[d].plot(np.arange(T), true_xs_i[..., d], color="black", label="Truth")
        
        

        ax[d].legend()
        ax[d].set_title(f"Dimension {d}")

    plt.savefig(f"{PLOTDIR}/smoothing_means_{args.i}.png")
    plt.close()


def plot_weight_diagnostics(log_ws):
    log_ws_i = log_ws[args.i]            # (M, T, N)
    log_w_norm = log_ws_i - logsumexp(log_ws_i, axis=2, keepdims=True)
    max_w = np.exp(np.max(log_w_norm, axis=2))       # (M, T), in (0,1]
    avg_max_ws = max_w.mean(axis=0)                   # (T,)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(avg_max_ws.shape[-1]), avg_max_ws)
    plt.savefig(f"{PLOTDIR}/weight_dominance_{args.i}.png")
    plt.close()


data = np.load(f"{dirpath}/data.npz")
plot_lineage(data["Bs"])
plot_ess(data["ess"])
plot_smoothing_means(data["true_xs"], data["xs"], data["ys"], data["inds"])
plot_weight_diagnostics(data["log_ws"])
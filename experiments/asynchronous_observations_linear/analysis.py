import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.asynchronous_observations_linear.kernels import KernelType
from gradient_csmc.utils.printing import ctext

from jax.scipy.special import logsumexp

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--D", dest="D", type=int, default=50)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=4)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=1_000)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=0)
parser.add_argument("--burnin", dest="burnin", type=int, default=3_000)

parser.add_argument("--target", dest="target", type=int, default=27)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.RW_CSMC)
parser.add_argument("--style", dest="style", type=str, default="na")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--i", type=int, default=0)
parser.add_argument("--j", type=int, default=0)

args = parser.parse_args()

TARGET_ALPHA = args.target / 100 
kernel_type = KernelType(args.kernel)

experiment_name = "kernel={},s={},b={},a={},M={},T={},D={},N={},style={},target={:.2f},res={},back={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.n_samples,
    args.burnin,
    args.adaptation,
    args.M, 
    args.D, 
    args.D, 
    args.N, 
    args.style, 
    TARGET_ALPHA, 
    args.resampling,
    args.backward,
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
    A = ancestors[args.i, :, args.j, ...]          # (N, T)
    x = np.arange(A.shape[-1])     # (T,)
    plt.figure(figsize=(15, 5))
    plt.plot(x, A.T, color="blue", alpha=0.1)
    plt.savefig(f"{PLOTDIR}/lineages_{args.i}_{args.j}.png")
    plt.close()


def plot_esjd(esjd):
    """
    esjd: (K, N-1, M, T)
    """
    esjd_i = esjd[args.i]              # (N-1, M, T)
    _, M, T = esjd_i.shape

    sample_mean = esjd_i.mean(axis=0)   # (M, T)
    chain_mean = sample_mean.mean(axis=0)
    chain_std = sample_mean.std(axis=0, ddof=1)

    ts = np.arange(T)

    plt.figure(figsize=(15, 5))
    plt.plot(ts, chain_mean, label="Mean across chains")
    plt.fill_between(ts, chain_mean - chain_std, chain_mean + chain_std, alpha=0.25, label="±1 std")
    plt.title("ESJD across chains (mean ± 1 std)")
    plt.xlabel("t")
    plt.ylabel("ESJD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/esjd_mean_std_{args.i}.png")
    plt.close()

    # plt.figure(figsize=(15, 5))
    # for m in range(M):
    #     plt.plot(np.arange(T), esjd_means[m, :], alpha=0.6, label=f"Chain {m}")
    # plt.title("Expected Square Jumping Distance")
    # plt.legend()
    # plt.savefig(f"{PLOTDIR}/esjd_{args.i}.png")    
    # plt.close()

def plot_ess(ess):
    ess_i = ess[args.i]                     # (N, M, T)
    chain_mean_ess = ess_i.mean(axis=1)     # (N, T)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(ess.shape[-1]), chain_mean_ess.T, alpha=0.01, color="black")
    plt.savefig(f"{PLOTDIR}/ess_{args.i}.png")
    plt.close()


def plot_smoothing_means(true_xs, xs, ys):
    true_xs_i = true_xs[args.i]   # (T, D)
    xs_i = xs[args.i]             # (M, T, D)
    ys_i = ys[args.i]             # (T, D)

    M, T, dx = xs_i.shape
    ts = np.arange(T)

    # chain_means = xs_i.mean(axis=0)
    
    dim = min(10, dx)
    fig, ax = plt.subplots(dim, 1, figsize=(15, 5*dim))
    ax = np.atleast_1d(ax)
    for d in range(dim):
        ys_d = ys_i[:, d]
        ax[d].plot(ts, ys_d, marker="x", color="red", label="Obs", alpha=0.3)

        # plot chains independently
        for m in range(M):
            ax[d].plot(ts, xs_i[m, :, d], label=f"Chain {m}")

        ax[d].plot(ts, true_xs_i[..., d], color="black", label="Truth")
        ax[d].legend()
        ax[d].set_title(f"Dimension {d}")

    plt.savefig(f"{PLOTDIR}/smoothing_means_{args.i}.png")
    plt.close()


def plot_weight_diagnostics(log_ws):
    log_ws_i = log_ws[args.i]            # (N, M, T, P)
    log_w_norm = log_ws_i - logsumexp(log_ws_i, axis=2, keepdims=True)
    max_w = np.exp(np.max(log_w_norm, axis=2))       # (M, T), in (0,1]
    avg_max_ws = max_w.mean(axis=0)                   # (T,)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(avg_max_ws.shape[-1]), avg_max_ws)
    plt.savefig(f"{PLOTDIR}/weight_dominance_{args.i}.png")
    plt.close()


data = np.load(f"{dirpath}/data.npz")
plot_lineage(data["Bs"])
plot_esjd(data["esjd_all"])
plot_ess(data["ess"])
plot_smoothing_means(data["true_xs"], data["means"], data["ys"])
# plot_weight_diagnostics(data["log_ws"])
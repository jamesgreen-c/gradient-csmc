import argparse
import os

from experiments.memory_efficient_svm.kernels import KernelType
from gradient_csmc.utils.plotting import (plot_ess, plot_rr_v_delta, 
                                          plot_traces, plot_xs, plot_square_error, plot_mae)

import matplotlib.pyplot as plt
import numpy as np


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=1_000)
parser.add_argument("--D", dest="D", type=int, default=100)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=3)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=25)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=2_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=1_000)

parser.add_argument("--target", dest="target", type=int, default=75)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, required=True)
parser.add_argument("--style", dest="style", type=str, required=True)

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")
parser.add_argument("--last-step", dest='last_step', type=str, default="forced")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--n-plot-states", dest="n_plot_states", type=int, default=5)
parser.add_argument("--i", type=int, default=0)


args = parser.parse_args()


kernel_type = KernelType(args.kernel)
TARGET_ALPHA = args.target / 100  # 1 - (1 + args.N) ** (-1 / 2)

experiment_name = "kernel={},s={},b={},a={},M={},T={},D={},N={},style={},target={:.2f},res={},back={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.n_samples,
    args.burnin,
    args.adaptation, 
    args.M, 
    args.T, 
    args.D, 
    args.N, 
    args.style, 
    TARGET_ALPHA, 
    args.resampling,
    args.backward,
    args.seed
)

# check path exists
dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    raise FileNotFoundError(
f"""

No such experiment found with the following configuration

Kernel: {kernel_type.name},
N Samples: {args.n_samples}, 
M: {args.M}, 
T: {args.T}, 
D: {args.D}, 
N Particles: {args.N}, 
style: {args.style}, 
Target Alpha: {TARGET_ALPHA}, 
Resampling Type: {args.resampling},
Backward Sampling: {args.backward},
Seed: {args.seed}
"""
    )

# Extract data from results
datapath = f"{dirpath}/data.npz"
data = np.load(datapath)

ess_all = data["ess"]
final_pct_all = data["final_pct"]
adapted_delta_all = data["delta"]
# sampling_time_all = data["sampling_time"]
# energy_all = data["energy"]
means_all = data["means"]
# std_devs_all = data["std_devs"]
true_xs_all = data["true_xs"]
ys_all = data["ys"]
# init_xs_all = data["init_xs"]
# acf_all = data["iacf_all"]
# esjd_all = data["esjd_all"]
# traces_all = data["traces_all"]

# make plot directory
plotdir = f"{dirpath}/plots"
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

def plot_smoothing_means(true_xs, init_xs, xs, ys, inds):
    true_xs_i = true_xs[args.i]
    init_xs_i = init_xs[args.i]
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
        
        ax[d].plot(np.arange(T), true_xs_i[..., d], color="black", label="Truth")
        ax[d].plot(np.arange(T), init_xs_i[..., d], color="green", label="CSMC", alpha=0.5)
        ax[d].plot(np.arange(T), chain_means[..., d], label="Mean")
        
        if ys_d.size > 0:
            ax[d].scatter(ts, ys_d, marker="x", color="red", label="Obs", alpha=1)

        ax[d].legend()
        ax[d].set_title(f"Dimension {d}")

    plt.savefig(f"{plotdir}/smoothing_means_{args.i}.png")
    plt.close()

# plot data
plot_rr_v_delta(final_pct_all, adapted_delta_all, plotdir)

n_plots = min(args.n_plot_states, true_xs_all.shape[-1])
plot_smoothing_means(data["true_xs"], data["init_xs"], data["means"], data["ys"], data["inds"])
plot_ess(ess_all, plotdir)
# plot_traces(traces_all, plotdir)
# plot_square_error(init_xs_all, true_xs_all, means_all, plotdir)
# plot_mae(init_xs_all, true_xs_all, means_all, plotdir)



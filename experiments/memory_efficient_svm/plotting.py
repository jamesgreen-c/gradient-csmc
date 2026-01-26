import argparse
import os

from experiments.scalable_factor_stoch_vol.kernels import KernelType
from gradient_csmc.utils.plotting import (plot_ess, plot_rr_v_delta, 
                                          plot_traces, plot_xs, plot_square_error, plot_mae)

import matplotlib.pyplot as plt
import numpy as np


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=128)
parser.add_argument("--D", dest="D", type=int, default=5)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--bpf-init", dest="bpf_init", action='store_true')
parser.add_argument('--no-bpf-init', dest='bpf_init', action='store_false')
parser.set_defaults(bpf_init=True)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=25)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=2000)
parser.add_argument("--burnin", dest="burnin", type=int, default=3000)

parser.add_argument("--target", dest="target", type=int, default=75)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, required=True)
parser.add_argument("--style", dest="style", type=str, required=True)

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="killing")
parser.add_argument("--last-step", dest='last_step', type=str, default="forced")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--n-plot-states", dest="n_plot_states", type=int, default=5)

args = parser.parse_args()


kernel_type = KernelType(args.kernel)
TARGET_ALPHA = args.target / 100  # 1 - (1 + args.N) ** (-1 / 2)

experiment_name = "kernel={},samples={},burnin={},adaptation={},M={},T={},D={},N={},style={},target={:.2f},bpf_init={},resampling={},backward={},seed={}"
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
    args.bpf_init,
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
BPF Init: {args.bpf_init},
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
sampling_time_all = data["sampling_time"]
energy_all = data["energy"]
means_all = data["means"]
std_devs_all = data["std_devs"]
true_xs_all = data["true_xs"]
ys_all = data["ys"]
init_xs_all = data["init_xs"]
acf_all = data["iacf_all"]
esjd_all = data["esjd_all"]
traces_all = data["traces_all"]

# make plot directory
plotdir = f"{dirpath}/plots"
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

# plot data
plot_rr_v_delta(final_pct_all, adapted_delta_all, plotdir)

n_plots = min(args.n_plot_states, true_xs_all.shape[-1])
plot_xs(init_xs_all, true_xs_all, means_all, std_devs_all, plotdir, components=list(range(n_plots)))
plot_ess(ess_all, plotdir)
plot_traces(traces_all, plotdir)
plot_square_error(init_xs_all, true_xs_all, means_all, plotdir)
plot_mae(init_xs_all, true_xs_all, means_all, plotdir)



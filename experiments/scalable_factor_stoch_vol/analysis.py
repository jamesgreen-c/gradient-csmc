import argparse
import os 

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from experiments.scalable_factor_stoch_vol.kernels import KernelType

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=128)
parser.add_argument("--D", dest="D", type=int, default=5)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--n-factors", dest="n_factors", type=int, default=5)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--bpf-init", dest="bpf_init", action='store_true')
parser.add_argument('--no-bpf-init', dest='bpf_init', action='store_false')
parser.set_defaults(bpf_init=True)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=25)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=2000)
parser.add_argument("--burnin", dest="burnin", type=int, default=3000)

parser.add_argument("--target", dest="target", type=int, default=75)

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="killing")
parser.add_argument("--last-step", dest='last_step', type=str, default="forced")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

args = parser.parse_args()


KERNELS = (
    KernelType.CSMC,
    KernelType.RW_CSMC,
    KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, KernelType.TP_CSMC, 
    KernelType.MALA_CSMC, KernelType.MALA_CSMC, KernelType.MALA_CSMC,
    KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC, KernelType.ADAPTED_CSMC,
)

STYLES = (
    'bootstrap',
    'na',
    'marginal', 'filtering', 'smoothing', 'twisted',
    'marginal', 'filtering', 'smoothing',
    'marginal', 'filtering', 'twisted',
)


TS = (args.T,)
TARGETS = (args.target,)

combination = list(product(TS, TARGETS, zip(KERNELS, STYLES)))


# ANALYSE DELTAS
def analyse_deltas():
    """
    We want to see how the adapted deltas change for varying methodologies (TP CSMC vs RW CSMC vs MALA CSMC) 
    while keeping the other parameters (eg N, T, D, F) fixed. Our setup here only accepts one parameter value
    so as to ensure any results loaded are matched in this regard.

    For each methodology, plot bars on 3 figures for each of the min, median and max deltas. 
    """

    stats = []
    for j in range(len(combination)):
        T, target, (kernel, style, *_) = combination[j]
        TARGET_ALPHA = target / 100

        experiment_name = (
            "kernel={},samples={},burnin={},adaptation={},M={},T={},D={},F={},N={},style={},target={:.2f},"
            "bpf_init={},resampling={},backward={},seed={}"
        ).format(
            kernel.name,
            args.n_samples,
            args.burnin,
            args.adaptation,
            args.M,
            T,
            args.D,
            args.n_factors,
            args.N,
            style,
            TARGET_ALPHA,
            args.bpf_init,
            args.resampling,
            args.backward,
            args.seed,
        )

        dirpath = f"results/{experiment_name}"
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"No such experiment found: {dirpath}")

        data = np.load(f"{dirpath}/data.npz")
        adapted_delta_all = data["delta"]  # (K, T)

        # Your current summary: average across K then summarise over time
        mean_adapted_delta = adapted_delta_all.mean(axis=0)  # (T,)

        stats.append(
            dict(
                kernel=kernel.name,
                style=str(style),
                dmin=float(np.min(mean_adapted_delta)),
                dmed=float(np.median(mean_adapted_delta)),
                dmax=float(np.max(mean_adapted_delta)),
            )
        )

    # ----------------- Plotting -----------------

    kernels = sorted({s["kernel"] for s in stats})
    styles = sorted({s["style"] for s in stats})

    # Build lookup: (kernel, style) -> (min, med, max)
    lookup = {(s["kernel"], s["style"]): (s["dmin"], s["dmed"], s["dmax"]) for s in stats}

    x = np.arange(len(kernels))
    n_styles = len(styles)
    group_width = 0.8
    bar_w = group_width / max(n_styles, 1)

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    titles = ["Min adapted delta (over time)", "Median adapted delta (over time)", "Max adapted delta (over time)"]
    keys = [0, 1, 2]  # index into (min, med, max)

    # Use a colormap so each style gets a consistent color across all panels
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_styles)]

    for si, style in enumerate(styles):
        # Center grouped bars around each kernel tick
        offset = (si - (n_styles - 1) / 2) * bar_w

        vals = np.array(
            [lookup.get((k, style), (np.nan, np.nan, np.nan)) for k in kernels],
            dtype=float,
        )  # shape (K, 3)

        for row, a in enumerate(ax):
            a.bar(x + offset, vals[:, row], width=bar_w * 0.95, label=style if row == 0 else None, color=colors[si])

    for i, a in enumerate(ax):
        a.set_title(titles[i])
        a.set_ylabel("delta")
        a.grid(True, axis="y", alpha=0.25)

    for a in ax:
        a.set_yscale("symlog", linthresh=1e-3)

    ax[-1].set_xticks(x)
    ax[-1].set_xticklabels(kernels, rotation=0)

    # One shared legend (much cleaner than repeating)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("results/delta_scaling.png")


if __name__ == "__main__":
    analyse_deltas()
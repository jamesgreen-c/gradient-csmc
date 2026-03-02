"""
Analyse ESJD with increasing N
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.asynchronous_observations_linear.kernels import KernelType
from gradient_csmc.utils.printing import ctext


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
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")

parser.add_argument("--i", type=int, default=0)
parser.add_argument("--j", type=int, default=0)

args = parser.parse_args()

TARGET_ALPHA = args.target / 100 
kernel_type = KernelType(args.kernel)
Ns = [31, 64, 128, 256, 512, 1024, 2048, 4096]

def load_data(N: int):
    """ Load data for a given number of particles N"""
    experiment_name = "kernel={},s={},b={},a={},M={},T={},D={},N={},style={},target={:.2f},res={},back={},seed={}"
    experiment_name = experiment_name.format(
        kernel_type.name,
        args.n_samples,
        args.burnin,
        args.adaptation,
        args.M, 
        args.D, 
        args.D, 
        N, 
        args.style, 
        TARGET_ALPHA, 
        args.resampling,
        args.backward,
        args.seed
    )

    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print(ctext("No such experiment exists", "yellow"))
        print(experiment_name)
        exit()

    data = np.load(f"{dirpath}/data.npz")
    return data


PLOTDIR = f"results/plots"
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


def plot_lineage():
    
    fig, ax = plt.subplots(len(Ns), 1, figsize=(15, 5*len(Ns)))
    for x, n in enumerate(Ns):
        data = load_data(n)
        ancestors = data["Bs"]    # (K, S, M, T)
        A_ij = ancestors[args.i, :, args.j, :]
        t = np.arange(A_ij.shape[-1])
        ax[x].plot(t, A_ij.T, color="blue", alpha=0.1)
        ax[x].set_title(f"Particles: {n}")
        ax[x].set_xlabel("Time")
        ax[x].set_ylabel("Ancestor Index")

    plt.savefig(f"{PLOTDIR}/lineages-by-particles.png")
    plt.close()


def plot_esjd():
    """
    esjd: (K, N-1, M, T)
    """
    mean_esjds = np.empty(len(Ns))
    std_esjds  = np.empty(len(Ns))

    for x, n in enumerate(Ns):
        data = load_data(n)
        esjd = data["esjd_all"]
        esjd_i = esjd[args.i]            # (S-1, M, T)

        # chain-wise means
        chain_means = esjd_i.mean(axis=(0, 2))  # mean over S-1 and T -> (M,)

        mean_esjds[x] = chain_means.mean()
        std_esjds[x]  = chain_means.std(ddof=1)

    Ns_arr = np.asarray(Ns)

    plt.figure(figsize=(15, 5))
    plt.plot(Ns_arr, mean_esjds, label="Mean across chains")
    plt.fill_between(Ns_arr, mean_esjds - std_esjds, mean_esjds + std_esjds,
                     alpha=0.25, label="±1 std (across chains)")
    plt.title("Mean ESJD by particle count N (mean ± 1 std across chains)")
    plt.xlabel("N")
    plt.ylabel("Mean ESJD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTDIR}/esjd-by-particles.png")
    plt.close()
    # mean_esjds = np.empty((len(Ns)))
    # for x, n in enumerate(Ns):
    #     data = load_data(n)
    #     esjd = data["esjd_all"]
    #     esjd_i = esjd[args.i]   # (S-1, M, T)
    #     mean_esjds[x] = esjd_i.mean()
    
    # plt.figure(figsize=(15, 5))
    # plt.plot(np.array(Ns), mean_esjds)
    # plt.title("Mean ESJD by particle count N")
    # plt.xlabel("N")
    # plt.ylabel("Mean ESJD")
    # plt.savefig(f"{PLOTDIR}/esjd-by-particles.png")    
    # plt.close()

plot_lineage()
plot_esjd()
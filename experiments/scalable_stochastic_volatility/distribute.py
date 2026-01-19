import argparse
import os 

from itertools import product

from experiments.scalable_stochastic_volatility.kernels import KernelType
from gradient_csmc.utils.printing import ctext

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=128)
parser.add_argument("--D", dest="D", type=int, default=5)

parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--bpf-init", dest="bpf_init", action='store_true')
parser.add_argument('--no-bpf-init', dest='bpf_init', action='store_false')
parser.set_defaults(bpf_init=True)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=25)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=2000)
parser.add_argument("--burnin", dest="burnin", type=int, default=3000)

parser.add_argument("--target", dest="target", type=int, default=75)

parser.add_argument("--i", dest="i", type=int, default=-1)

parser.add_argument("--plot", dest="plot", action="store_true")
parser.set_defaults(plot=False)

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

N = args.N
D = args.D

if args.i == -1:
    for j in range(len(combination)):
        T, target, (kernel, style, *_) = combination[j]
        exec_str = (
            "python3 experiment.py "
            "--target {} --T {} --kernel {} --style {} --D {} "
            "--N {} --adaptation {} --burnin {} --n-samples {}"
        )
        exec_str = exec_str if args.bpf_init else f"{exec_str} --no-bpf-init"
        exec_str = exec_str.format(target, T, kernel.value, style, D, N, args.adaptation, args.burnin, args.n_samples)
        print("\nExcecuting: ", ctext(exec_str, "green"))
        os.system(exec_str)

        if args.plot:
            plotting_str = (
                "python3 plotting.py "
                "--target {} --T {} --kernel {} --style {} --D {} "
                "--N {} --adaptation {} --burnin {} --n-samples {}"
            )
            plotting_str = plotting_str if args.bpf_init else f"{plotting_str} --no-bpf-init"
            plotting_str = plotting_str.format(target, T, kernel.value, style, D, N, args.adaptation, args.burnin, args.n_samples)
            print("\nPlotting: ", ctext(plotting_str, "green"))
            os.system(plotting_str)

else:
    T, target, (kernel, style, *_) = combination[args.i]
    exec_str = (
        "python3 experiment.py "
        "--target {} --T {} --kernel {} --style {} --D {} "
        "--N {} --adaptation {} --burnin {} --n-samples {}"
    )
    exec_str = exec_str if args.bpf_init else f"{exec_str} --no-bpf-init"
    exec_str = exec_str.format(target, T, kernel.value, style, D, N, args.adaptation, args.burnin, args.n_samples)
    print("\nExcecuting: ", ctext(exec_str, "green"))
    os.system(exec_str)

    if args.plot:
        plotting_str = (
            "python3 plotting.py "
            "--target {} --T {} --kernel {} --style {} --D {} "
            "--N {} --adaptation {} --burnin {} --n-samples {}"
        )
        plotting_str = plotting_str if args.bpf_init else f"{plotting_str} --no-bpf-init"
        plotting_str = plotting_str.format(target, T, kernel.value, style, D, N, args.adaptation, args.burnin, args.n_samples)
        print("\nPlotting: ", ctext(plotting_str, "green"))
        os.system(plotting_str)
        



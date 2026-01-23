import argparse
import os 

from itertools import product

from experiments.scalable_factor_stoch_vol.kernels import KernelType
from gradient_csmc.utils.printing import ctext

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--bpf-init", dest="bpf_init", action='store_true')
parser.add_argument('--no-bpf-init', dest='bpf_init', action='store_false')
parser.set_defaults(bpf_init=True)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=5_000)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=1000)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.MALA)
parser.add_argument("--style", dest="style", type=str, default="auxiliary")

parser.add_argument("--i", dest="i", type=int, default=-1)

parser.add_argument("--plot", dest="plot", action="store_true")
parser.set_defaults(plot=False)

args = parser.parse_args()

kernel_type = (KernelType(args.kernel), )
styles = (args.style, )

TS = (64, 128, 256)
DS = (100,)
FS = (10, 20, 30)
NS = (31, 255, 511)

TARGETS = (25, 50, 75)

combination = list(product(kernel_type, styles, TS, DS, FS, NS, TARGETS))

def build_cmd(script: str, kernel, style, T, D, F, N, target) -> str:
    base = (
        f"python3 {script} "
        f"--target {target} --T {T} --kernel {kernel.value} --style {style} --D {D} "
        f"--n-factors {F} --N {N} --adaptation {args.adaptation} --burnin {args.burnin} --n-samples {args.n_samples}"
    )
    return base if args.bpf_init else f"{base} --no-bpf-init"

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    kernel, style, T, D, F, N, target = combination[j]
    exec_str = build_cmd("experiment.py", kernel, style, T, D, F, N, target)
    print("\nExecuting:", ctext(exec_str, "green"))
    os.system(exec_str)

    if args.plot:
        plotting_str = build_cmd("plotting.py", kernel, style, T, D, F, N, target)
        print("\nPlotting:", ctext(plotting_str, "green"))
        os.system(plotting_str)



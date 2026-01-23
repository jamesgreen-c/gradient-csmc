import argparse
import os 

from itertools import product

from experiments.scalable_factor_stoch_vol.kernels import KernelType
from gradient_csmc.utils.printing import ctext

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--M", dest="M", type=int, default=1)

parser.add_argument("--bpf-init", dest="bpf_init", action='store_true')
parser.add_argument('--no-bpf-init', dest='bpf_init', action='store_false')
parser.set_defaults(bpf_init=True)

parser.add_argument("--n-samples", dest="n_samples", type=int, default=5_000)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=10_000)
parser.add_argument("--burnin", dest="burnin", type=int, default=1000)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.MALA)
parser.add_argument("--style", dest="style", type=str, default="auxiliary")

parser.add_argument("--resampling", dest='resampling', type=str, default="killing")
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--i", dest="i", type=int, default=-1)

parser.add_argument("--plot", dest="plot", action="store_true")
parser.set_defaults(plot=False)

args = parser.parse_args()


def results_exist(*, kernel, style, T, D, N, target, args) -> bool:
    """
    Mirror experiment.py's experiment_name + datapath convention and check if results already exist.
    """
    TARGET_ALPHA = target / 100  # must match experiment.py convention

    experiment_name = (
        "kernel={},samples={},burnin={},adaptation={},M={},T={},D={},N={},style={},target={:.2f},"
        "bpf_init={},resampling={},backward={},seed={}"
    ).format(
        kernel.name,   # kernel_type.name in experiment.py
        args.n_samples,
        args.burnin,
        args.adaptation,
        args.M,
        T,
        D,
        N,
        style,
        TARGET_ALPHA,
        args.bpf_init,
        args.resampling,
        args.backward,
        args.seed,
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


kernel_type = (KernelType(args.kernel), )
styles = (args.style, )

TS = (64, 128, 256)
DS = (50, 100, 150)
NS = (31, 255, 511)

TARGETS = (25, 50, 75)

combination = list(product(kernel_type, styles, TS, DS, NS, TARGETS))

def build_cmd(script: str, kernel, style, T, D, N, target) -> str:
    base = (
        f"python3 {script} "
        f"--target {target} --T {T} --kernel {kernel.value} --style {style} --D {D} "
        f"--N {N} --adaptation {args.adaptation} --burnin {args.burnin} --n-samples {args.n_samples}"
    )
    return base if args.bpf_init else f"{base} --no-bpf-init"

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    kernel, style, T, D, N, target = combination[j]
    
    if results_exist(kernel=kernel, style=style, T=T, D=D, N=N, target=target, args=args):
        print(ctext(f"Skipping (already run): T={T}, D={D}, N={N}, target={target}", "yellow"))
        continue  # or return / pass depending on your structure
    
    exec_str = build_cmd("experiment.py", kernel, style, T, D, N, target)
    print("\nExecuting:", ctext(exec_str, "green"))
    # os.system(exec_str)

    if args.plot:
        plotting_str = build_cmd("plotting.py", kernel, style, T, D, N, target)
        print("Plotting:", ctext(plotting_str, "green"))
        # os.system(plotting_str)



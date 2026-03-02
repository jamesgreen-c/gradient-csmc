import argparse
import os 

from itertools import product

from experiments.scalable_factor_stoch_vol.kernels import KernelType
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
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

args = parser.parse_args()

TARGET_ALPHA = args.target / 100 
kernel_type = KernelType(args.kernel)
Ns = (31, 64, 128, 256, 512, 1024, 2048, 4096,)


print(f"""
######### CONFIGURATION #########
-- Kernel:     {kernel_type.name}
-- Style:      {args.style}

-- Samples:    {args.n_samples}
-- Burnin:     {args.burnin}
-- Adaptation: {args.adaptation}

-- M:          {args.M}
-- T:          {args.D}
-- D:          {args.D}

-- Resampling: {args.resampling}
-- Backward:   {args.backward}
-- Seed:       {args.seed}
#################################
""")

TARGET_ALPHA = args.target / 100  # must match experiment.py convention

def results_exist(*, N) -> bool:
    """
    Mirror experiment.py's experiment_name + datapath convention and check if results already exist.
    """

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

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


def build_cmd(script: str, N) -> str:
    cmd = (
        f"python3 {script} "
        f"--target {args.target} --kernel {kernel_type.value} --style {args.style} --D {args.D} "
        f"--N {N} --adaptation {args.adaptation} --burnin {args.burnin} --n-samples {args.n_samples} "
        f"--M {args.M} --K {args.K} "
    )
    return cmd 

for n in Ns:
    
    if results_exist(N=n):
        print(ctext(f"Skipping (already run): N={n}", "yellow"))
        continue
    
    exec_str = build_cmd(script="experiment.py", N=n)
    print("\nExecuting:", ctext(exec_str, "green"))
    os.system(exec_str)


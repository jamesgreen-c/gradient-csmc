import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from functools import partial

from experiments.asynchronous_observations.kernels import KernelType
from experiments.asynchronous_observations.model import get_data
from gradient_csmc.utils.common import force_move, barker_move, ess
from gradient_csmc.utils.kalman import sampling, filtering
from gradient_csmc.utils.resamplings import killing, multinomial

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=1_000)
parser.add_argument("--D", dest="D", type=int, default=10)
parser.add_argument("--K", dest="K", type=int, default=50)
parser.add_argument("--M", dest="M", type=int, default=100)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--delta", dest="delta", type=float,
                    default=1.)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1 / 3)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="na")
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")
parser.add_argument("--last-step", dest='last_step', type=str, default="barker")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

parser.add_argument("--plot", action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.set_defaults(plot=False)

args = parser.parse_args()

print(f"""
###############################################
#    ASYNCHRONOUS OBSERVATIONS EXPERIMENT     #
###############################################
Configuration:
    - T: {args.T}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
""")

# BACKEND CONFIG
NOW = time.time()

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)

if args.delta_arg == "D":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "T":
    DELTA = args.delta / args.T ** args.delta_scale
elif args.delta_arg == "DT" or args.delta_arg == "TD":
    DELTA = args.delta / (args.D * args.T) ** args.delta_scale
else:
    DELTA = args.delta

if args.resampling == "killing":
    resampling_fn = killing
elif args.resampling == "multinomial":
    resampling_fn = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.last_step == "forced":
    last_step_fn = force_move
elif args.last_step == "barker":
    last_step_fn = barker_move
else:
    raise ValueError(f"Unknown last step {args.last_step}")

kernel_type = KernelType(args.kernel)
DELTA = kernel_type.shape_delta(DELTA, args.T)
SIGMA = 10 ** (args.log_var / 2)


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, sample_key = jax.random.split(key, 3)

    true_xs, ys, inds = get_data(data_key, SIGMA, args.D, args.T)
    # jax.debug.print("True xs shape = {}, ys shape = {}, inds shape = {}", true_xs.shape, ys.shape, inds.shape)

    kernel_, init, = kernel_type.kernel_maker(ys, inds, args.D, SIGMA, N=args.N,
                                              resampling_func=resampling_fn,
                                              backward=args.backward,
                                              ancestor_move_func=last_step_fn,
                                              style=args.style,
                                              conditional=False)

    kernel_ = jax.jit(kernel_)

    # Since conditional=False, x_star is only used for shape extraction.
    init_xs = jnp.zeros(shape=true_xs.shape)
    init_state = init(init_xs)

    sample_keys = jax.random.split(sample_key, args.M)

    def get_sample(k_):
        next_xs, next_Bs, next_log_ws, *_ = kernel_(k_, init_state, DELTA)
        return (next_xs, next_Bs, next_log_ws)

    with jax.disable_jit(args.debug):
        all_xs, all_Bs, all_log_ws = jax.vmap(get_sample)(sample_keys)

    all_ess = jax.vmap(partial(ess, log_weights=True))(all_log_ws)  # (M, )
    return all_xs, all_Bs, all_log_ws, all_ess, true_xs, ys, inds


xs_all = np.empty((args.K, args.M, args.T, args.D))
Bs_all = np.empty((args.K, args.M, args.T))
log_ws_all = np.empty((args.K, args.M, args.T, args.N+1))
ess_all = np.empty((args.K, args.M, args.T))
true_xs_all = np.empty((args.K, args.T, args.D))
ys_all = np.empty((args.K, args.T))
inds_all = np.empty((args.K, args.T))


for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")

    tic = time.time()
    (xs_k, Bs_k, log_ws_k, ess_k, true_xs_k, ys_k, inds_k) = one_experiment(key_k)

    toc = time.time()
    sample_time_k = (toc - tic) / args.M
    
    xs_all[k, ...] = xs_k
    Bs_all[k, ...] = Bs_k
    log_ws_all[k, ...] = log_ws_k
    ess_all[k, ...] = ess_k
    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    inds_all[k, ...] = inds_k

    print(f"""
Results:
    - sampling time: {float(sample_time_k):.0f}s
    - final average ess: {np.mean(ess_k):.2f} 
""")
    print()

if not os.path.exists("results"):
    os.mkdir("results")

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
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath,
    xs=xs_all,
    Bs=Bs_all,
    log_ws=log_ws_all,
    ess=ess_all,
    true_xs=true_xs_all,
    ys=ys_all, 
    inds=inds_all
)

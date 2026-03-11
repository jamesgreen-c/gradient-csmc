import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from experiments.asynchronous_observations.kernels import KernelType, get_csmc_kernel
from experiments.asynchronous_observations.model import get_data
from gradient_csmc.utils.common import force_move, barker_move, ess
from gradient_csmc.utils.resamplings import killing, multinomial, dynamic

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

# Adaption config
MIN_DELTA = 1e-7
MAX_DELTA = 1e2
MIN_RATE = 1e-3
ADAPTATION_WINDOW = 100
ADAPTATION_RATE = 0.85

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--D", dest="D", type=int, default=50)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=4)

parser.add_argument("--log-var", dest="log_var", type=float, default=-1.5)
parser.add_argument("--phi", type=float, default=0.9)

parser.add_argument("--delta", dest="delta", type=float,
                    default=1.)
parser.add_argument("--delta-scale", dest="delta_scale", type=float, default=1 / 3)
parser.add_argument("--delta-arg", dest="delta_arg", type=str, default="na")

parser.add_argument("--n-samples", dest="n_samples", type=int, default=1_000)
parser.add_argument("--adaptation", dest="adaptation", type=int, default=0)
parser.add_argument("--burnin", dest="burnin", type=int, default=3_000)
parser.add_argument("--delta-init", dest="delta_init", type=float,
                     default=10 ** (0.5 * (np.log10(MIN_DELTA) + np.log10(MAX_DELTA))))

parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5)
parser.set_defaults(dynamic=False)

parser.add_argument("--target", dest="target", type=int, default=27)
parser.add_argument("--target-stat", dest='target_stat', type=str, default="mean")

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

KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)
N_LAGS = min(args.n_samples - 1, 1_000)


print(f"""
###############################################
#    ASYNCHRONOUS OBSERVATIONS EXPERIMENT     #
###############################################
Configuration:
    - T: {args.D}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
    - N (Particles): {args.N+1}
""")

# BACKEND CONFIG
NOW = time.time()


TARGET_ALPHA = args.target / 100  # 1 - (1 + args.N) ** (-1 / 2)
if args.target_stat.isnumeric():
    TARGET_STAT = float(args.target_stat) / 100
else:
    TARGET_STAT = args.target_stat

if args.delta_arg == "D":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "T":
    DELTA = args.delta / args.D ** args.delta_scale
elif args.delta_arg == "DT" or args.delta_arg == "TD":
    DELTA = args.delta / (args.D * args.D) ** args.delta_scale
else:
    DELTA = args.delta

if args.resampling == "killing":
    resampling_func = killing
elif args.resampling == "multinomial":
    resampling_func = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.dynamic:
    assert args.threshold is not None, "If using dynamic sampling, please provide a threshold for the ESS"
    def resampling_fn(key, weights, i, j, conditional):
        return dynamic(resampling_func, args.threshold, key, weights, i, j, conditional)
else:
    resampling_fn = resampling_func

if args.last_step == "forced":
    last_step_fn = force_move
elif args.last_step == "barker":
    last_step_fn = barker_move
else:
    raise ValueError(f"Unknown last step {args.last_step}")

kernel_type = KernelType(args.kernel)
DELTA = kernel_type.shape_delta(DELTA, args.D)
SIGMA = 10 ** (args.log_var / 2)
SIGMA_Y = 0.1
PHI = args.phi


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, init_key, adaptation_key, burnin_key, sample_key = jax.random.split(key, 5)

    true_xs, m0, ys, inds, chol_P0, chol_Q = get_data(data_key, args.D, SIGMA, SIGMA_Y, PHI)
    # jax.debug.print("True xs shape = {}, ys shape = {}, inds shape = {}", true_xs.shape, ys.shape, inds.shape)

    kernel, init, adaptation_loop, experiment_loop = kernel_type.kernel_maker(m0, ys, inds, args.D, PHI, chol_P0, chol_Q, SIGMA_Y, N=args.N,
                                                                              resampling_func=resampling_fn,
                                                                              backward=args.backward,
                                                                              ancestor_move_func=last_step_fn,
                                                                              style=args.style)

    adaptation_kernel = kernel
    kernel = jax.jit(kernel)
    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 5, 6), static_argnames=("window_size", "target_stat"))
    experiment_loop = jax.jit(experiment_loop, static_argnums=(2, 3, 4, 5))

    csmc_kernel, csmc_init, *_ = get_csmc_kernel(m0, ys, inds, args.D, PHI, chol_P0, chol_Q, SIGMA_Y, N=args.N, resampling_func=resampling_fn,
                                                 backward=True, ancestor_move_func=None, conditional=False)
    
    # This looks like it's using the true data, but it's not (see, the conditional=False above)
    # We only pass it for the shape of the data.
    init_xs, *_ = csmc_kernel(init_key, csmc_init(true_xs), None)
    init_state = init(init_xs)

    with jax.disable_jit(args.debug):
        adaptation_state, adapted_delta = adaptation_loop(adaptation_key, init_state, adaptation_kernel,
                                                          TARGET_ALPHA,
                                                          args.delta_init, args.adaptation, args.verbose,
                                                          min_delta=MIN_DELTA, max_delta=MAX_DELTA,
                                                          window_size=ADAPTATION_WINDOW,
                                                          rate=ADAPTATION_RATE, min_rate=MIN_RATE,
                                                          target_stat=TARGET_STAT,
                                                          )

    if args.verbose:
        jax.debug.print("Adaptation delta median = {}, min = {}, max = {}", jnp.median(adapted_delta),
                        jnp.min(adapted_delta), jnp.max(adapted_delta))

    delta_kernel = lambda k_, s: kernel(k_, s, adapted_delta)
    burnin_keys = jax.random.split(burnin_key, args.M)
    sample_keys = jax.random.split(sample_key, args.M)

    def get_samples(sample_key_op, init_state_op, all_samples, n_samples):
        return experiment_loop(sample_key_op, init_state_op, delta_kernel, n_samples,
                               args.verbose, all_samples)

    with jax.disable_jit(args.debug):
        burnin_sample_xs, burnin_pct = jax.vmap(get_samples, in_axes=[0, None, None, None], out_axes=0)(burnin_keys,
                                                                                                      adaptation_state,
                                                                                                      False,
                                                                                                      args.burnin)

    burnin_states = jax.vmap(init, in_axes=0)(burnin_sample_xs)

    sample_xs, sample_bs, sample_log_ws, final_pct = jax.vmap(get_samples, in_axes=[0, 0, None, None], out_axes=1)(sample_keys, 
                                                                                       burnin_states, 
                                                                                       True,
                                                                                       args.n_samples)

    
    final_pct = final_pct.astype(jnp.float32).mean(axis=0)   # (M*T,) or (something,)
    final_pct = final_pct.reshape(args.M, args.D)            # (M, T)

    # energy = log_pdf(sample_xs, ys, inds, SIGMA)
    sample_ess = jax.vmap(jax.vmap(partial(ess, log_weights=True)))(sample_log_ws)  # (M, )
    # jax.debug.print("ESS shape = {}, log ws shape = {}", sample_ess.shape, sample_log_ws.shape)

    means_here = jnp.mean(sample_xs, 0)
    std_devs_here = jnp.std(sample_xs, 0)

    esjd = jnp.sum(
        (sample_xs[1:] - sample_xs[:-1]) ** 2, -1
    ) 

    # extract traces for representative dimensions
    # t_idx = jnp.array([0, args.D // 2, args.D - 1])
    # traces = jnp.take(samples[:, :, :, 0], t_idx, axis=2)

    return (means_here, std_devs_here, sample_ess, final_pct, esjd,
            # energy, 
            init_xs, true_xs, ys, adapted_delta, sample_bs, inds)


final_pct_all = np.empty((args.K, args.M, args.D))
ess_all = np.empty((args.K, args.n_samples, args.M, args.D))
adapted_delta_all = np.empty((args.K, args.D))
means_all = np.empty((args.K, args.M, args.D, args.D))
std_devs_all = np.empty((args.K, args.M, args.D, args.D))
true_xs_all = np.empty((args.K, args.D, args.D))
ys_all = np.empty((args.K, args.D))
init_xs_all = np.empty((args.K, args.D, args.D))
esjd_all = np.empty((args.K, args.n_samples - 1, args.M, args.D))
Bs_all = np.empty((args.K, args.n_samples, args.M, args.D))
# log_ws_all = np.empty((args.K, args.n_samples, args.M, args.D, args.N+1))
inds_all = np.empty((args.K, args.D))
# energy_all = np.empty((args.K, args.n_samples, args.M))


for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")

    tic = time.time()
    (means_k, std_k, ess_k, final_pct_k, esjd_vals_k,
     # energy_k,
     init_xs_k, true_xs_k, ys_k, adapted_delta_k, Bs_k, inds_k) = one_experiment(key_k)
    toc = time.time()
    sample_time_k = (toc - tic) / args.M

    
    final_pct_all[k, ...] = final_pct_k
    ess_all[k, :, :] = np.asarray(ess_k)
    saved_delta_k = adapted_delta_k * np.ones((args.D,))
    adapted_delta_all[k, :] = np.asarray(saved_delta_k)
    means_all[k, ...] = means_k
    std_devs_all[k, ...] = std_k
    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    init_xs_all[k, ...] = init_xs_k
    esjd_all[k, ...] = np.asarray(esjd_vals_k)
    Bs_all[k, ...] = Bs_k
    # log_ws_all[k, ...] = log_ws_k
    inds_all[k, ...] = inds_k
    # energy_all[k, :] = np.asarray(energy_k)


    print(f"""
Results:
    - sampling time: {float(sample_time_k):.0f}s
    - final min-max acceptance rate: {np.min(final_pct_k):.2%}, {np.max(final_pct_k):.2%}
    - final min-max delta: {np.min(saved_delta_k):.2E}, {np.max(saved_delta_k):.2E}
    - final min-max ess: {np.min(ess_k):.2f}, {np.max(ess_k):.2f} 
    - final argmin-argmax ess: {np.argmin(ess_k)}, {np.argmax(ess_k)}
""")
    print()
    

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},s={},b={},a={},M={},T={},D={},N={},style={},target={:.2f},res={},back={},dynamic={},seed={}"
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
    args.dynamic,
    args.seed
)


dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(datapath, 
                    means=means_all, std_devs=std_devs_all, final_pct=final_pct_all, ess=ess_all,
                    esjd_all=esjd_all, 
                    init_xs=init_xs_all, true_xs=true_xs_all, ys=ys_all,
                    delta=adapted_delta_all,
                    inds=inds_all, Bs=Bs_all)
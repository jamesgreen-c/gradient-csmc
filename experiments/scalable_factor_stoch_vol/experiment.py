import argparse
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from experiments.scalable_factor_stoch_vol.kernels import KernelType, get_csmc_kernel
from experiments.scalable_factor_stoch_vol.model import get_dynamics, get_data, log_pdf
from gradient_csmc.utils.common import force_move, barker_move
from gradient_csmc.utils.resamplings import killing, multinomial

PLOTDIR = "plots"

# Adaption config
MIN_DELTA = 1e-11
MAX_DELTA = 1e2
MIN_RATE = 1e-3
ADAPTATION_WINDOW = 100
ADAPTATION_RATE = 0.85

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

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
parser.add_argument("--delta-init", dest="delta_init", type=float,
                     default=10 ** (0.5 * (np.log10(MIN_DELTA) + np.log10(MAX_DELTA))))
# default=1.0)
parser.add_argument("--target", dest="target", type=int, default=75)
parser.add_argument("--target-stat", dest='target_stat', type=str, default="mean")

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.MALA)
parser.add_argument("--style", dest="style", type=str, default="auxiliary")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="killing")
parser.add_argument("--last-step", dest='last_step', type=str, default="forced")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--verbose", action='store_true')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)


args = parser.parse_args()


# GENERAL CONFIG
DIM = (args.n_factors * (args.n_factors-1) // 2) + args.n_factors
KEY = jax.random.PRNGKey(args.seed)
EXPERIMENT_KEYS = jax.random.split(KEY, args.K)
N_LAGS = min(args.n_samples - 1, 1_000)


print(f"""
###############################################
#  SCALABLE STOCHASTIC VOLATILITY EXPERIMENT  #
###############################################
Configuration:
    - T: {args.T}
    - target: {args.target}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
    - F: {args.n_factors}
    - Latent State Dimension: {DIM}
    - BPF Init: {args.bpf_init}
""")

# BACKEND CONFIG
NOW = time.time()

# we use the parametrisation given by ...
m0, P0, Q, F, b, B, V = get_dynamics(args.D, args.n_factors)

TARGET_ALPHA = args.target / 100  # 1 - (1 + args.N) ** (-1 / 2)
if args.target_stat.isnumeric():
    TARGET_STAT = float(args.target_stat) / 100
else:
    TARGET_STAT = args.target_stat

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


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    """
    Runs a single experiment to find the ESJD of latent states under a given CSMC sampling kernel (eg Particle-RWM), 
    and given settings (eg number of particles, data length, data dimension, dynamics parameters).

    Gets the data from user defined parameters. 
    
    Uses kernel_maker to generate: 
     1. The MCMC kernel function (eg Particle-RWM kernel takes a reference trajectory and returns an updated one)
     2. The adaption loop (Runs delta adaptation over multiple MCMC steps)
     3. The experiment loop (Runs MCMC sampling over multiple steps, collecting samples)

    It uses a single CSMC kernel to generate a BPF initialization of the reference trajectory.
    It then finds the optimal deltas via the adaptation loop. It fixes these deltas by constructing the delta kernel. 
    Defines a get_samples function that runs the experiment loop to get samples, if all_samples is FALSE then only the 
    final state is returned. 

    Runs M independent chains of: 
     1. Burn-in sampling to get burn-in samples (not stored)
     2. Sampling to get samples (stored) - uses burn-in final states as initials.

     Calculates statistics:
      1. ESS of the samples
      2. Auto-correlation function of the samples
      3. Means and standard deviations of the samples (M, T, D) -> averages over N samples per chain 
    
    """
    data_key, init_key, adaptation_key, burnin_key, sample_key = jax.random.split(key, 5)

    true_xs, true_fs, ys, inv_chol_P0, inv_chol_Q = get_data(data_key, m0, P0, Q, F, b, B, V, args.n_factors, args.T)

    kernel, init, adaptation_loop, experiment_loop = kernel_type.kernel_maker(ys, m0, P0, Q, F, b, B, V, args.N,
                                                                              resampling_func=resampling_fn,
                                                                              backward=args.backward,
                                                                              ancestor_move_func=last_step_fn,
                                                                              style=args.style)

    adaptation_kernel = kernel
    kernel = jax.jit(kernel)
    adaptation_loop = jax.jit(adaptation_loop, static_argnums=(2, 5, 6), static_argnames=("window_size", "target_stat"))
    experiment_loop = jax.jit(experiment_loop, static_argnums=(2, 3, 4, 5))

    csmc_kernel, csmc_init, *_ = get_csmc_kernel(ys, m0, P0, Q, F, b, B, V, N=args.N, resampling_func=resampling_fn,
                                                 backward=True, ancestor_move_func=None, conditional=False)

    if args.bpf_init:
        # This looks like it's using the true data, but it's not (see, the conditional=False above)
        # We only pass it for the shape of the data.
        init_xs, *_ = csmc_kernel(init_key, csmc_init(true_xs), None)
        # jax.debug.print("Init xs shape = {}", init_xs.shape)
    else:
        init_xs = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=(args.T, DIM)
        ) * 2 + 1.5
    
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
        burnin_samples, burnin_pct = jax.vmap(get_samples, in_axes=[0, None, None, None], out_axes=0)(burnin_keys,
                                                                                                      adaptation_state if args.bpf_init else init_state,
                                                                                                      False,
                                                                                                      args.burnin)

    burnin_states = jax.vmap(init, in_axes=0)(burnin_samples)

    samples, final_pct = jax.vmap(get_samples, in_axes=[0, 0, None, None], out_axes=1)(sample_keys, 
                                                                                       burnin_states, 
                                                                                       True,
                                                                                       args.n_samples)

    # jax.debug.print("samples shape: {}", samples.shape)
    final_pct = final_pct.astype(jnp.float32).mean(axis=0)   # (M*T,) or (something,)
    final_pct = final_pct.reshape(args.M, args.T)            # (M, T)
    # final_pct = jnp.mean(final_pct * 1.0, 0)
    # final_pct = jnp.reshape(final_pct, (args.M, -1)) * jnp.ones((args.M, args.T))
    # jax.debug.print("final_pct shape: {}", final_pct.shape)
    energy = log_pdf(samples, ys, m0, inv_chol_P0, F, b, inv_chol_Q, B, V)

    if args.M > 1:
        samples_ess = tfp.mcmc.effective_sample_size(samples, filter_beyond_positive_pairs=False,
                                                     cross_chain_dims=1) / args.M
    else:
        samples_ess = tfp.mcmc.effective_sample_size(samples[:, 0, ...], filter_beyond_positive_pairs=False)

    samples_acfs = tfp.stats.auto_correlation(samples, axis=0, max_lags=N_LAGS)
    samples_acf = jnp.mean(samples_acfs, 1)

    means_here = jnp.mean(samples, 0)
    std_devs_here = jnp.std(samples, 0)

    def esjd(xs, xs_next):
        return jnp.sum((xs - xs_next) ** 2, -1)

    with jax.disable_jit(args.debug):
        if args.M > 1:
            esjd_vals = jax.vmap(
                lambda s: jax.vmap(esjd)(s[:-1], s[1:])  # vmap over adjacent pairs in time
            )(samples)  # vmap over M chains
        else:
            esjd_vals = jax.vmap(esjd)(samples[:-1], samples[1:])  


    # extract traces for representative dimensions
    # traces = (samples[:, :, 0, 0], samples[:, :, args.T // 2, 0], samples[:, :, args.T - 1, 0])
    t_idx = jnp.array([0, args.T // 2, args.T - 1])
    traces = jnp.take(samples[:, :, :, 0], t_idx, axis=2)

    return (means_here, std_devs_here, samples_ess, final_pct,
            energy, init_xs, true_xs, ys, adapted_delta, 
            samples_acf, esjd_vals, traces)


final_pct_all = np.empty((args.K, args.M, args.T))
ess_all = np.empty((args.K, args.T, DIM))
energy_all = np.empty((args.K, args.n_samples, args.M))
adapted_delta_all = np.empty((args.K, args.T))
sampling_time_all = np.empty((args.K,))
means_all = np.empty((args.K, args.M, args.T, DIM))
std_devs_all = np.empty((args.K, args.M, args.T, DIM))
true_xs_all = np.empty((args.K, args.T, DIM))
ys_all = np.empty((args.K, args.T, args.D))
init_xs_all = np.empty((args.K, args.T, DIM))
acf_all = np.empty((args.K, N_LAGS + 1, args.T, DIM))
esjd_all = np.empty((args.K, args.n_samples - 1, args.M, args.T))
traces_all = np.empty((args.K, args.n_samples, args.M, 3))


for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")

    tic = time.time()
    (means_k, std_k, ess_k, final_pct_k,
     energy_k, init_xs_k, true_xs_k, ys_k, 
     adapted_delta_k, samples_iacf_k,
     esjd_vals_k, traces_k) = one_experiment(key_k)
    # print(ess_k.shape)

    toc = time.time()
    sample_time_k = (toc - tic) / args.M

    
    final_pct_all[k, ...] = final_pct_k
    ess_all[k, :, :] = np.asarray(ess_k)
    energy_all[k, :] = np.asarray(energy_k)
    saved_delta_k = adapted_delta_k * np.ones((args.T,))
    adapted_delta_all[k, :] = np.asarray(saved_delta_k)
    sampling_time_all[k] = sample_time_k
    means_all[k, ...] = means_k
    std_devs_all[k, ...] = std_k
    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    init_xs_all[k, ...] = init_xs_k
    acf_all[k, ...] = samples_iacf_k
    esjd_all[k, ...] = np.asarray(esjd_vals_k)
    traces_all[k, ...] = np.asarray(traces_k)

    # print("final_pct_k shape: ", final_pct_k.shape)
    # print("ess_k shape: ", np.asarray(ess_k).shape)
    # print("energy_k shape: ", np.asarray(energy_k).shape)
    # print("saved_delta_k shape: ", np.asarray(saved_delta_k).shape)
    # print("sample_time_k shape: ", sample_time_k.shape)
    # print("means_k shape: ", means_k.shape)
    # print("std_k shape: ", std_k.shape)
    # print("true_xs_k shape: ", true_xs_k.shape)
    # print("ys_k shape: ", ys_k.shape)
    # print("init_xs_k shape: ", init_xs_k.shape)
    # print("samples_iacf_k shape: ", samples_iacf_k.shape)

    print(f"""
Results:
    - sampling time: {float(sample_time_k):.0f}s
    - final min-max acceptance rate: {np.min(final_pct_k):.2%}, {np.max(final_pct_k):.2%}
    - final min-max delta: {np.min(saved_delta_k):.2E}, {np.max(saved_delta_k):.2E}
    - final min-max ess: {np.min(ess_k):.2f}, {np.max(ess_k):.2f} 
    - final argmin-argmax ess: {np.argmin(ess_k)}, {np.argmax(ess_k)}
    - final min-max energy: {np.min(energy_k):.2E}, {np.max(energy_k):.2E}
""")
    print()
    

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},samples={},burnin={},adaptation={},M={},T={},D={},F={},N={},style={},target={:.2f},bpf_init={},resampling={},backward={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.n_samples,
    args.burnin,
    args.adaptation,
    args.M, 
    args.T, 
    args.D, 
    args.n_factors, 
    args.N, 
    args.style, 
    TARGET_ALPHA, 
    args.bpf_init,
    args.resampling,
    args.backward,
    args.seed
)


dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(datapath, ess=ess_all, final_pct=final_pct_all, delta=adapted_delta_all,
                    sampling_time=sampling_time_all, energy=energy_all, means=means_all, std_devs=std_devs_all,
                    true_xs=true_xs_all, ys=ys_all, init_xs=init_xs_all, iacf_all=acf_all,
                    esjd_all=esjd_all, traces_all=traces_all)

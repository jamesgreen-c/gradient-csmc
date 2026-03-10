"""

INDEPENDENT FILTERING EXPERIMENT

To find the exact filtering distribution at observation time t_i for dimension i,
we sample from the prior dynamics up to t_{i-1} and then use a single step "particle filter"
that weights proposals to give us an empirical estimate of the filtering distribution.
We run this for a high number of particles. 

"""

import argparse
import os
import time
from typing import Callable, Union, Any

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

import numpy as np

from functools import partial

from experiments.asynchronous_observations.kernels import KernelType, get_bpf_kernel
from experiments.asynchronous_observations.model import get_data, log_potential

from gradient_csmc.utils.common import force_move, barker_move, ess
from gradient_csmc.utils.resamplings import killing, multinomial, dynamic
from gradient_csmc.utils.resamplings import normalize
from gradient_csmc.utils.math import mvn_logpdf
from gradient_csmc.utils.prior import sample as prior_sample

from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp


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

print(f"""
####################################################################
#    ASYNCHRONOUS OBSERVATIONS INDEPENDENT FILTERING EXPERIMENT    #
####################################################################
Configuration:
    - T: {args.D}
    - kernel: {KernelType(args.kernel).name}
    - style: {args.style}
    - D: {args.D}
    - N (Particles): {args.N+1}
""")


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
    # resampling_fn = jax.jit(closure)
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


def Ft(
        key: PRNGKey,
        x_t: Array,
        x_t_m_1: Array,
        M_t_logpdf: Callable,
        Gamma_t: Callable,
        y_t: Array,
    ):
    """
    Return a Normal approximation to the empirical distribution returned by a single weight calculation step 

    Params:
    x_t:        Latent (scalar) at time of observation.
    x_t_m_1:    Latent (scalar) at t minus 1
    M_t_logpdf: Log PDF function to calculate x_t | x_t_m_1
    Gamma_t:    M*G
    N:          Number of particles to use (N+1, if we include the reference trajectory).
    """

    # Compute weights and normalize
    log_wt = Gamma_t(x_t_m_1, x_t, y_t) - M_t_logpdf(x_t_m_1, x_t, y_t)
    log_wt = normalize(log_wt, log_space=True)
    wt = jnp.exp(log_wt)

    # resample
    A_t = resampling_fn(key, wt, None, None, False)
    x_t = jnp.take(x_t, A_t, axis=0)
    wt = jnp.take(wt, A_t, axis=0)
    wt = normalize(jnp.log(wt), log_space=False)

    # jax.debug.print("wt min {}, wt max {}", wt.min(), wt.max())
    # jax.debug.print("xt min {}, xt max {}", x_t.min(), x_t.max())

    # use wt and x_t to approximate a normal distribution: tuple(mean, std)
    x_t = jnp.ravel(x_t)
    wt = jnp.ravel(wt)

    mean = jnp.sum(wt * x_t)
    var = jnp.sum(wt * (x_t - mean) ** 2)
    std = jnp.sqrt(var)
    return jnp.array([mean, std])

def F0(
        key: PRNGKey,
        x_0: Array,
        M_0_logpdf: Callable,
        Gamma_0: Callable,
    ):
    """
    Return a Normal approximation to the empirical distribution returned by a single weight calculation step 

    Params:
    x_0:        Latent (scalar) at time of observation.
    M_0_logpdf: Log PDF function to calculate x_0 | m_0
    Gamma_0:    M*G
    N:          Number of particles to use (N+1, if we include the reference trajectory).
    """

    # Compute weights and normalize
    log_w0 = Gamma_0(x_0) - M_0_logpdf(x_0)
    log_w0 = normalize(log_w0, log_space=True)
    w0 = jnp.exp(log_w0)

    # resample
    A_t = resampling_fn(key, w0, None, None, False)
    x_0 = jnp.take(x_0, A_t, axis=0)
    w0 = jnp.take(w0, A_t, axis=0)
    w0 = normalize(jnp.log(w0), log_space=False)

    # use w0 and x_0 to approximate a normal distribution: tuple(mean, std)
    x_0 = jnp.ravel(x_0)
    w0 = jnp.ravel(w0)

    mean = jnp.sum(w0 * x_0)
    var = jnp.sum(w0 * (x_0 - mean) ** 2)
    std = jnp.sqrt(var)
    return jnp.array([mean, std])


@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    """

    """
    data_key, init_key, *_ = jax.random.split(key, 5)

    true_xs, m0, ys, inds, chol_P0, chol_Q = get_data(data_key, args.D, SIGMA, SIGMA_Y, PHI)
    # jax.debug.print("True xs shape = {}, ys shape = {}, inds shape = {}", true_xs.shape, ys.shape, inds.shape)
    
    dx = args.D
    dt = 1 / dx
    A = jnp.exp(-PHI * dt)

    def M0_rvs(_key, _):
        eps = jax.random.normal(_key, (args.N, 1))
        return 0 + (eps * SIGMA)

    def Mt_rvs(_key, x_t_m_1, _):
        eps = jax.random.normal(_key, (args.N, 1))
        return A * x_t_m_1 + (eps * SIGMA)

    M0_logpdf = lambda x: norm.logpdf(x, loc=0, scale=SIGMA).sum()
    M0_logpdf = jnp.vectorize(M0_logpdf, signature="(d)->()")
    Mt_logpdf = lambda x_t_m_1, x_t, _params: norm.logpdf(x_t, loc=A*x_t_m_1, scale=SIGMA).sum()
    Mt_logpdf = jnp.vectorize(Mt_logpdf, signature="(d),(d)->()", excluded=(2,))
    Gamma_0 = lambda x: log_potential(x, ys[0], 0, SIGMA_Y) + M0_logpdf(x)
    Gamma_t = lambda x_t_m_1, x_t, _params: log_potential(x_t, _params, 0, SIGMA_Y) + Mt_logpdf(x_t_m_1, x_t, None)

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, ys[1:]

    def prior_sampler(key_t):
        return prior_sample(key_t, M0, Mt, args.N, args.D, True)
    prior_sampler = jax.jit(prior_sampler)

    def get_distributions(_key):
        """ 
        1. Approximate distribution for x_0 (no loop required)
        2. Approximate distribution for x_t for t in [1, T] 
        """
        
        def body(_, inp):
            key_t, t, y_t = inp
            key_t_sample, key_t_res = jax.random.split(key_t, 2)
            xs = prior_sampler(key_t_sample)
            dist_x_t = Ft(key_t_res, xs[t], xs[t-1], Mt_logpdf, Gamma_t, y_t)
            return None, dist_x_t
        
        key_0_rvs, key_0_res, _key = jax.random.split(_key, 3)
        x_0 = M0_rvs(key_0_rvs, ys[0])
        dist_x_0 = F0(key_0_res, x_0, M0_logpdf, Gamma_0)
        
        # run loop
        keys_loop = jax.random.split(_key, dx - 1)
        inputs = keys_loop, jnp.arange(1, dx), ys[1:]
        _, dists = jax.lax.scan(body, None, inputs)
        dists = jnp.insert(dists, 0, dist_x_0, axis=0)
        return dists
    
    dists_ = get_distributions(key)

    #################################################
    #   Run single BPF to compare independent PFs   #
    # ###############################################
    bpf_kernel, bpf_init, *_ = get_bpf_kernel(m0, ys, inds, args.D, PHI, chol_P0, chol_Q, SIGMA_Y,
                                              N=args.N, 
                                              resampling_func=resampling_fn,
                                              conditional=False)
    _, _, _, _, log_ws, xs = bpf_kernel(init_key, bpf_init(true_xs), None)
    # jax.debug.print("log ws shape: {}, xs shape: {}", log_ws.shape, xs.shape)

    # calculate filter distribution
    log_ws = log_ws - logsumexp(log_ws, axis=1, keepdims=True)
    ws = jnp.exp(log_ws)
    means = jnp.sum(ws[:, :, None] * xs, axis=1)
    vars_ = jnp.sum(ws[:, :, None] * (xs - means[:, None, :]) ** 2, axis=1)
    stds = jnp.sqrt(vars_)
    joint_dists = jnp.stack([means, stds], axis=-1)
    # jax.debug.print("joint dists shape: {}",  joint_dists.shape)

    return true_xs, dists_, ys, joint_dists


true_xs_all = np.empty((args.K, args.D, args.D))
dists_all = np.empty((args.K, args.D, 2))
ys_all = np.empty((args.K, args.D))
joint_dists_all = np.empty((args.K, args.D, args.D, 2))

for k, key_k in enumerate(EXPERIMENT_KEYS):
    print(f"Running experiment {k + 1}/{args.K}")
    true_xs_k, dists_k, ys_k, joint_dists_k = one_experiment(key_k)

    true_xs_all[k, ...] = true_xs_k
    ys_all[k, ...] = ys_k
    dists_all[k, ...] = dists_k
    joint_dists_all[k, ...] = joint_dists_k

    print(f"""
    - True Xs (k) shape: {true_xs_k.shape}
    - Ys (k) shape: {ys_k.shape}
    - Dists (k) shape: {dists_k.shape}
    - Joint Dists (k): {joint_dists_k.shape}
""")
    print()

if not os.path.exists("if-results"):
    os.mkdir("if-results")

experiment_name = "D={},N={},dynamic={},seed={}"
experiment_name = experiment_name.format(
    args.D,
    args.N,
    args.dynamic,
    args.seed
)

dirpath = f"if-results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath, 
    true_xs=true_xs_all,
    ys=ys_all,
    dists=dists_all,
    joint_dists=joint_dists_all
)
"""
Implementation of kernel constructors for the scalable stochastic volatility 
model from Tistias 2023.
"""

from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, block_diag

import gradient_csmc.al_csmc_f as alf
import gradient_csmc.al_csmc_s as als
import gradient_csmc.atp_csmc_f as atpf
import gradient_csmc.atp_csmc_s as atps
import gradient_csmc.csmc as csmc
import gradient_csmc.l_csmc_f as lf
import gradient_csmc.mala as mala
import gradient_csmc.rw_csmc as rw
import gradient_csmc.t_atp_csmc_f as t_atpf
import gradient_csmc.tp as tp
import gradient_csmc.tp_csmc as tpf

from gradient_csmc.utils.math import mvn_logpdf
from gradient_csmc.utils.diag_mvn import diag_mvn_logpdf
from gradient_csmc.utils.mcmc_utils import sampling_routine, delta_adaptation_routine

from experiments.memory_efficient_svm.model import log_likelihood, log_potential, log_pdf


class KernelType(Enum):
    CSMC = 0
    TP = 1
    TP_CSMC = 2
    MALA_CSMC = 3
    ADAPTED_CSMC = 4
    RW_CSMC = 5
    MALA = 6

    @property
    def kernel_maker(self):
        if self == KernelType.TP:
            return get_tp_kernel
        elif self == KernelType.CSMC:
            return get_csmc_kernel
        elif self == KernelType.TP_CSMC:
            return get_tp_csmc_kernel
        elif self == KernelType.MALA_CSMC:
            return get_mala_csmc_kernel
        elif self == KernelType.ADAPTED_CSMC:
            return partial(get_tp_csmc_kernel, stop_gradient=True)
        elif self == KernelType.RW_CSMC:
            return get_rw_csmc_kernel
        elif self == KernelType.MALA:
            return get_mala_kernel
        else:
            raise NotImplementedError


def get_tp_kernel(ys, m0, P0, F, Q, b, N=1, **_kwargs):
    raise(NotImplementedError)


def get_mala_kernel(ys, m0, P0, F, Q, b, N=1, style="marginal", **_kwargs):
    raise(NotImplementedError)


def get_csmc_kernel(ys, m0, P0_diag, sigma, phi, b, N, style="bootstrap", **kwargs):
    """
    Implement the CSMC kernel for the scalable factor stochastic volatility model.
    Constructs the proposal distributions Mo and Mt as well as the potential functions
    Gamma_0 and Gamma_t.

    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0_diag: Vector of stdevs for the independet Gaussian latent states
    :param sigma: Vector of stdev for independent Gaussian latent states
    :param phi: Vector of persistence parameters for independent latent states
    :param b: Transition bias for the latent states
    :param N: Number of particles
    :param style: Style of CSMC kernel ('bootstrap' supported)
    :param kwargs: Additional keyword arguments
    """
    dx = m0.shape[0]

    # chol_P0 = jnp.linalg.cholesky(P0)
    # chol_Q = jnp.linalg.cholesky(Q)

    # _chol_P0_inv = solve_triangular(chol_P0, jnp.eye(m0.shape[0]), lower=True)
    # _chol_Q_inv = solve_triangular(chol_Q, jnp.eye(m0.shape[0]), lower=True)

    if style == "bootstrap":
        def M0_rvs(key, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return m0[None, ...] + sigma * eps

        def Mt_rvs(key, x_t_m_1, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return phi * x_t_m_1 + b[None, ...] + sigma * eps

        M0_logpdf = lambda x: diag_mvn_logpdf(x, m0, P0_diag, constant=False)
        Mt_logpdf = lambda x_t_m_1, x_t, _params: diag_mvn_logpdf(x_t, phi * x_t_m_1 + b[None, ...],
                                                                   sigma, constant=False)
        # M0_logpdf = lambda x: mvn_logpdf(x, m0, None, chol_inv=_chol_P0_inv, constant=False)
        # Mt_logpdf = lambda x_t_m_1, x_t, _params: mvn_logpdf(x_t, x_t_m_1 @ F.T + b[None, ...], None,
        #                                                      chol_inv=_chol_Q_inv, constant=False)
        Gamma_0 = lambda x: log_potential(x, ys[0]) + M0_logpdf(x)
        Gamma_t = lambda x_t_m_1, x_t, y: log_potential(x_t, y) + Mt_logpdf(x_t_m_1, x_t, None)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, ys[1:]
    Gamma_t_plus_params = Gamma_t, ys[1:]

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        return samples, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **_kwargs):
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **_kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn



def get_tp_csmc_kernel(ys, m0, P0_diag, sigma, phi, b, N, style="marginal", stop_gradient=False, **kwargs):
    """
    Implement the Twisted Particle-mGRAD kernel for the scalable stochastic volatility model.

    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0_diag: Vector of stdevs for the independet Gaussian latent states
    :param sigma: Vector of stdev for independent Gaussian latent states
    :param phi: Vector of persistence parameters for independent latent states
    :param b: Transition bias for the latent states
    :param kwargs: Additional keyword arguments
    """

    def r0(x):
        if stop_gradient:
            return log_potential(jax.lax.stop_gradient(x), ys[0])
        return log_potential(x, ys[0])

    def rt(_, x, y):
        if stop_gradient:
            return log_potential(jax.lax.stop_gradient(x), y)
        return log_potential(x, y)

    rt_plus_params = rt, ys[1:]

    mut = lambda x, _: phi * x + b[None, ...]
    
    
    raise(NotImplementedError)
    # below this shows how all code is written for matrix calculations that are memory intensive


    # Qs = jnp.repeat(Q[None, ...], ys.shape[0] - 1, axis=0)
    # if style == 'marginal':
    #     kernel = tpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    # elif style == 'filtering':
    #     kernel = atpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    # elif style == 'smoothing':
    #     kernel = atps.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    # elif style == 'twisted':
    #     Fs = jnp.repeat(F[None, ...], ys.shape[0] - 1, axis=0)
    #     bs = jnp.repeat(b[None, ...], ys.shape[0] - 1, axis=0)
    #     kernel = t_atpf.get_kernel(m0, P0, r0, Fs, bs, Qs, rt_plus_params, N=N, **kwargs)
    # else:
    #     raise NotImplementedError(
    #         f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing', 'twisted'")

    # wrapped_kernel = lambda key, state, delta: kernel(key, state[0], state[1], delta, delta)

    # init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    # def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
    #     samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
    #     return samples, flags

    # def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
    #                        n_steps, verbose, **kwargs_):
    #     if style == "twisted":
    #         return t_atpf.delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
    #                                                initial_delta, n_steps, verbose, **kwargs_)
    #     return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
    #                                     initial_delta, n_steps, verbose, **kwargs_)

    # return wrapped_kernel, init, adaptation_routine, sampling_routine_fn


def get_mala_csmc_kernel(ys, m0, P0_diag, sigma, phi, b, N, style="marginal", **kwargs):
    """
    Implement the Particle-aMALA algorithm for the scalable stochastic volatility model.
    
    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0_diag: Vector of stdevs for the independet Gaussian latent states
    :param sigma: Vector of stdev for independent Gaussian latent states
    :param phi: Vector of persistence parameters for independent latent states
    :param b: Transition bias for the latent states
    :param kwargs: Additional keyword arguments
    """
    raise(NotImplementedError)
    # @partial(jnp.vectorize, signature='(d)->()')
    # def Gamma_0(x):
    #     return log_potential(x, ys[0]) + diag_mvn_logpdf(x, m0, P0_diag, constant=False)

    # @partial(jnp.vectorize, signature='(d),(d),(n)->()')
    # def Gamma_t(x_t_m_1, x_t, y):
    #     x_pred = phi * x_t_m_1 + b
    #     return log_potential(x_t, y) + diag_mvn_logpdf(x_t, x_pred, sigma, constant=False)

    # Gamma_t_plus_params = Gamma_t, ys[1:]

    # if style == "filtering":
    #     kernel = lambda key, state, delta: alf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
    #                                                   delta, N=N, **kwargs)
    # elif style == "smoothing":
    #     kernel = lambda key, state, delta: als.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
    #                                                   delta, N=N, **kwargs)
    # elif style == "marginal":
    #     kernel = lambda key, state, delta: lf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
    #                                                  delta, N=N, **kwargs)
    # else:
    #     raise NotImplementedError(f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing'")
    # init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    # def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
    #     samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
    #     return samples, flags

    # def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
    #                        n_steps, verbose, **_kwargs):
    #     return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
    #                                     initial_delta, n_steps, verbose, **_kwargs)

    # return kernel, init, adaptation_routine, sampling_routine_fn


def get_rw_csmc_kernel(ys, m0, P0_diag, sigma, phi, b, N, **kwargs):
    """
    Implement the random-walk CSMC kernel for the scalable stochastic volatility model.

    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0: Initial covariances for the latent states
    :param Q: Process noise covariances for the latent states
    :param F: Persistence coefficients for h and d
    :param b: Bias for the latent states
    :param kwargs: Additional keyword arguments
    """

    kwargs.pop("style")
    # dx = m0.shape[0]

    # chol_P0 = jnp.linalg.cholesky(P0)
    # chol_Q = jnp.linalg.cholesky(Q)

    # chol_P0_inv = solve_triangular(chol_P0, jnp.eye(dx), lower=True)
    # chol_Q_inv = solve_triangular(chol_Q, jnp.eye(dx), lower=True)

    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0]) + diag_mvn_logpdf(x, m0, P0_diag, constant=False)

    @partial(jnp.vectorize, signature='(d),(d),(n)->()')
    def Gamma_t(x_t_m_1, x_t, y):
        x_pred = phi * x_t_m_1 + b
        return log_potential(x_t, y) + diag_mvn_logpdf(x_t, x_pred, sigma, constant=False)

    Gamma_t_plus_params = Gamma_t, ys[1:]

    kernel = lambda key, state, delta: rw.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params,
                                                 delta, N=N, **kwargs)

    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        return samples, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **_kwargs):
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **_kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn


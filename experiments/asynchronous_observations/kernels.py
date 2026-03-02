from enum import Enum
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import gradient_csmc.csmc as csmc
import gradient_csmc.rw_csmc as rw
from experiments.asynchronous_observations.model import log_potential
from gradient_csmc.utils.mcmc_utils import aux_sampling_routine, delta_adaptation_routine
from gradient_csmc.utils.math import mvn_logpdf
from jax.scipy.linalg import solve_triangular


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

    def shape_delta(self, delta, T):
        if self == KernelType.TP:
            return delta
        elif self == KernelType.CSMC:
            return delta
        elif self == KernelType.TP_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.MALA_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.ADAPTED_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.RW_CSMC:
            return delta * np.ones((T,))
        elif self == KernelType.RW:
            return delta
        elif self == KernelType.IMH:
            return delta
        elif self == KernelType.MALA:
            return delta
        
#######################
# Kernel constructors #
#######################


def get_tp_kernel(ys, m0, P0, F, Q, b, N=1, **_kwargs):
    raise NotImplementedError


def get_mala_kernel(ys, m0, P0, F, Q, b, N=1, style="marginal", **_kwargs):
    raise NotImplementedError


def get_csmc_kernel(m0, ys, inds, dx, phi, chol_P0, chol_Q, N, style="bootstrap", **kwargs):

    dt = 1 / dx
    A = jnp.exp(phi * dt)

    _chol_P0_inv = solve_triangular(chol_P0, jnp.eye(m0.shape[0]), lower=True)
    _chol_Q_inv = solve_triangular(chol_Q, jnp.eye(m0.shape[0]), lower=True)

    if style == "bootstrap":
        def M0_rvs(key, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return eps @ chol_P0.T

        def Mt_rvs(key, x_t_m_1, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return A * x_t_m_1 + eps @ chol_Q.T
            # return x_t_m_1 + sigma * eps

        # M0_logpdf = lambda x: norm.logpdf(x, scale=sigma).sum()
        # Mt_logpdf = lambda x_t_m_1, x_t, _params: norm.logpdf(x_t, x_t_m_1, sigma).sum()
        M0_logpdf = lambda x: mvn_logpdf(x, m0, None, chol_inv=_chol_P0_inv, constant=False)
        M0_logpdf = jnp.vectorize(M0_logpdf, signature="(d)->()")
        Mt_logpdf = lambda x_t_m_1, x_t, _params: mvn_logpdf(x_t, A * x_t_m_1, None, chol_inv=_chol_Q_inv, constant=False).sum()
        Mt_logpdf = jnp.vectorize(Mt_logpdf, signature="(d),(d)->()", excluded=(2,))
        Gamma_0 = lambda x: log_potential(x, ys[0], inds[0]) + M0_logpdf(x)
        Gamma_t = lambda x_t_m_1, x_t, _params: log_potential(x_t, _params[0], _params[1]) + Mt_logpdf(x_t_m_1, x_t, None)

    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, ys[1:]
    Gamma_t_plus_params = Gamma_t, (ys[1:], inds[1:])

    kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
                                                **kwargs)
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        return aux_sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        # return sample_xs, sample_bs, sample_log_ws, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **_kwargs):
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **_kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn


def get_tp_csmc_kernel(ys, m0, P0, F, Q, b, N, style="marginal", stop_gradient=False, **kwargs):
    raise NotImplementedError


def get_mala_csmc_kernel(ys, m0, P0, F, Q, b, N, style="marginal", **kwargs):
    raise NotImplementedError


def get_rw_csmc_kernel(m0, ys, inds, dx, phi, chol_P0, chol_Q, N, **kwargs):
    kwargs.pop("style")
    dt = 1 / dx
    A = jnp.exp(phi * dt)

    _chol_P0_inv = solve_triangular(chol_P0, jnp.eye(m0.shape[0]), lower=True)
    _chol_Q_inv = solve_triangular(chol_Q, jnp.eye(m0.shape[0]), lower=True)

    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0], inds[0]) + mvn_logpdf(x, m0, None, chol_inv=_chol_P0_inv, constant=False)
        # return log_potential(x, ys[0], inds[0]) + norm.logpdf(x, scale=sigma).sum()

    @partial(jnp.vectorize, signature='(d),(d),(k)->()')
    def Gamma_t(x_t_m_1, x_t, _params):
        x_pred = A * x_t_m_1
        return log_potential(x_t, _params[0], _params[1]) + mvn_logpdf(x_t, x_pred, None, chol_inv=_chol_Q_inv, constant=False)
        # Q_k = sigma * jnp.sqrt((jnp.exp(2 * phi * dt) - 1) / (2 * phi))
        # return log_potential(x_t, _params[0], _params[1]) + norm.logpdf(x_t, x_t_m_1, scale=Q_k).sum()

    Gamma_t_plus_params = Gamma_t, (ys[1:], inds[1:])

    kernel = lambda key, state, delta: rw.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params,
                                                 delta, N=N, **kwargs)

    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        sample_xs, sample_bs, sample_log_ws, flags = aux_sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        return sample_xs, sample_bs, sample_log_ws, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **_kwargs):
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **_kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn
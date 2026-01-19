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
from gradient_csmc.utils.mcmc_utils import sampling_routine, delta_adaptation_routine

from experiments.scalable_stochastic_volatility.model import log_likelihood, log_potential, log_pdf
from experiments.scalable_stochastic_volatility.utils import _check_param_shapes


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


def get_csmc_kernel(ys, m0, P0, Q, F, b, N, style="bootstrap", **kwargs):
    """
    Implement the CSMC kernel for the scalable factor stochastic volatility model.
    Constructs the proposal distributions Mo and Mt as well as the potential functions
    Gamma_0 and Gamma_t.

    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0: Initial covariances for the latent states
    :param Q: Process noise covariances for the latent states
    :param F: Transition matrix for the latent states
    :param b: Transition bias for the latent states
    :param N: Number of particles
    :param style: Style of CSMC kernel ('bootstrap' supported)
    :param kwargs: Additional keyword arguments
    """
    dx = m0.shape[0]

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    _chol_P0_inv = solve_triangular(chol_P0, jnp.eye(m0.shape[0]), lower=True)
    _chol_Q_inv = solve_triangular(chol_Q, jnp.eye(m0.shape[0]), lower=True)

    if style == "bootstrap":
        def M0_rvs(key, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return m0[None, ...] + eps @ chol_P0.T

        def Mt_rvs(key, x_t_m_1, _):
            eps = jax.random.normal(key, (N + 1, dx))
            return x_t_m_1 @ F.T + eps @ chol_Q.T + b[None, ...]

        M0_logpdf = lambda x: mvn_logpdf(x, m0, None, chol_inv=_chol_P0_inv, constant=False)
        Mt_logpdf = lambda x_t_m_1, x_t, _params: mvn_logpdf(x_t, x_t_m_1 @ F.T + b[None, ...], None,
                                                             chol_inv=_chol_Q_inv, constant=False)
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



def get_tp_csmc_kernel(ys, m0, P0, Q, F, b, N, style="marginal", stop_gradient=False, **kwargs):
    """
    Implement the Twisted Particle-mGRAD kernel for the scalable stochastic volatility model.

    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0: Initial covariances for the latent states
    :param Q: Process noise covariances for the latent states
    :param F: Persistence coefficients for h and d
    :param b: Bias for the latent states
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

    mut = lambda x, _: x @ F.T + b[None, ...]

    Qs = jnp.repeat(Q[None, ...], ys.shape[0] - 1, axis=0)
    if style == 'marginal':
        kernel = tpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'filtering':
        kernel = atpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'smoothing':
        kernel = atps.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
    elif style == 'twisted':
        Fs = jnp.repeat(F[None, ...], ys.shape[0] - 1, axis=0)
        bs = jnp.repeat(b[None, ...], ys.shape[0] - 1, axis=0)
        kernel = t_atpf.get_kernel(m0, P0, r0, Fs, bs, Qs, rt_plus_params, N=N, **kwargs)
    else:
        raise NotImplementedError(
            f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing', 'twisted'")

    wrapped_kernel = lambda key, state, delta: kernel(key, state[0], state[1], delta, delta)

    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        return samples, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **kwargs_):
        if style == "twisted":
            return t_atpf.delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                                   initial_delta, n_steps, verbose, **kwargs_)
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **kwargs_)

    return wrapped_kernel, init, adaptation_routine, sampling_routine_fn


def get_mala_csmc_kernel(ys, m0, P0, Q, F, b, N, style="marginal", **kwargs):
    """
    Implement the Particle-aMALA algorithm for the scalable stochastic volatility model.
    
    :param ys: Observations (returns)
    :param m0: Initial means for the latent states
    :param P0: Initial covariances for the latent states
    :param Q: Process noise covariances for the latent states
    :param F: Persistence coefficients for h and d
    :param b: Bias for the latent states
    :param kwargs: Additional keyword arguments
    """

    dx = m0.shape[0]

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    chol_P0_inv = solve_triangular(chol_P0, jnp.eye(dx), lower=True)
    chol_Q_inv = solve_triangular(chol_Q, jnp.eye(dx), lower=True)

    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0]) + mvn_logpdf(x, m0, None, chol_inv=chol_P0_inv, constant=False)

    @partial(jnp.vectorize, signature='(d),(d),(n)->()')
    def Gamma_t(x_t_m_1, x_t, y):
        x_pred = x_t_m_1 @ F.T + b
        return log_potential(x_t, y) + mvn_logpdf(x_t, x_pred, None, chol_inv=chol_Q_inv, constant=False)

    Gamma_t_plus_params = Gamma_t, ys[1:]

    if style == "filtering":
        kernel = lambda key, state, delta: alf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                      delta, N=N, **kwargs)
    elif style == "smoothing":
        kernel = lambda key, state, delta: als.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                      delta, N=N, **kwargs)
    elif style == "marginal":
        kernel = lambda key, state, delta: lf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
                                                     delta, N=N, **kwargs)
    else:
        raise NotImplementedError(f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing'")
    init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

    def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
        samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
        return samples, flags

    def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
                           n_steps, verbose, **_kwargs):
        return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
                                        initial_delta, n_steps, verbose, **_kwargs)

    return kernel, init, adaptation_routine, sampling_routine_fn


def get_rw_csmc_kernel(ys, m0, P0, Q, F, b, N, **kwargs):
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
    dx = m0.shape[0]

    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Q = jnp.linalg.cholesky(Q)

    chol_P0_inv = solve_triangular(chol_P0, jnp.eye(dx), lower=True)
    chol_Q_inv = solve_triangular(chol_Q, jnp.eye(dx), lower=True)

    @partial(jnp.vectorize, signature='(d)->()')
    def Gamma_0(x):
        return log_potential(x, ys[0]) + mvn_logpdf(x, m0, None, chol_inv=chol_P0_inv, constant=False)

    @partial(jnp.vectorize, signature='(d),(d),(n)->()')
    def Gamma_t(x_t_m_1, x_t, y):
        x_pred = x_t_m_1 @ F.T + b
        return log_potential(x_t, y) + mvn_logpdf(x_t, x_pred, None, chol_inv=chol_Q_inv, constant=False)

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



# ============================ OLD ===================

# def get_csmc_kernel(ys, m0, P0, Q, phi, N, style="bootstrap", **kwargs):
#     """
#     Implement the CSMC kernel for the scalable stochastic volatility model.
#     Constructs the proposal distributions Mo and Mt as well as the potential functions
#     Gamma_0 and Gamma_t.

#     :param ys: Observations (returns)
#     :param m0: (h0, d0) Initial means for the latent states
#     :param P0: (h_P, d_P) Initial covariances for the latent states
#     :param Q: (Sigma_h, Sigma_d) Process noise covariances for the latent states
#     :param phi: (phi_h, phi_d) Persistence coefficients for h and d
#     :param N: Number of particles
#     :param style: Style of CSMC kernel ('bootstrap' supported)
#     :param kwargs: Additional keyword arguments
#     """

#     _check_param_shapes(m0, P0, Q, phi)

#     h0, d0 = m0
#     dh, dd = h0.shape[0], d0.shape[0]
#     phi_h, phi_d = phi
#     Sigma_h, Sigma_d = Q

#     if Sigma_h.ndim == 1:
#         Sigma_h = jnp.diag(Sigma_h)
#     if Sigma_d.ndim == 1:
#         Sigma_d = jnp.diag(Sigma_d)

#     h_P0, d_P0 = P0

#     chol_h_P0 = jnp.linalg.cholesky(h_P0)
#     chol_d_P0 = jnp.linalg.cholesky(d_P0)
#     chol_Sigma_h = jnp.linalg.cholesky(Sigma_h)
#     chol_Sigma_d = jnp.linalg.cholesky(Sigma_d)

#     _inv_chol_h_P0 = solve_triangular(chol_h_P0, jnp.eye(h0.shape[0]), lower=True)
#     _inv_chol_d_P0 = solve_triangular(chol_d_P0, jnp.eye(d0.shape[0]), lower=True)
#     _inv_chol_Sigma_h = solve_triangular(chol_Sigma_h, jnp.eye(h0.shape[0]), lower=True)
#     _inv_chol_Sigma_d = solve_triangular(chol_Sigma_d, jnp.eye(d0.shape[0]), lower=True)

#     if style == "bootstrap":
#         def M0_rvs(key, _):
#             key_h, key_d = jax.random.split(key, 2)
#             eps_h = jax.random.normal(key_h, (N + 1, dh))
#             eps_d = jax.random.normal(key_d, (N + 1, dd))

#             h1 = h0[None, ...] + eps_h @ chol_h_P0.T
#             d1 = d0[None, ...] + eps_d @ chol_d_P0.T
#             return jnp.concatenate([h1, d1], axis=-1)

#         def Mt_rvs(key, x_t_m_1, _):
#             key_h, key_d = jax.random.split(key, 2)
#             eps_h = jax.random.normal(key_h, (N + 1, dh))
#             eps_d = jax.random.normal(key_d, (N + 1, dd))

#             h_t_m_1 = x_t_m_1[:, :dh]
#             d_t_m_1 = x_t_m_1[:, dh:]

#             h_t = h0 + phi_h * (h_t_m_1 - h0) + eps_h @ chol_Sigma_h.T
#             d_t = d0 + phi_d * (d_t_m_1 - d0) + eps_d @ chol_Sigma_d.T
#             return jnp.concatenate([h_t, d_t], axis=-1)

#         def M0_logpdf(x):
#             x = jnp.atleast_2d(x)
#             hs = x[:, :dh]
#             ds = x[:, dh:]

#             logp_h = mvn_logpdf(hs, h0, None, chol_inv=_inv_chol_h_P0, constant=False)
#             logp_d = mvn_logpdf(ds, d0, None, chol_inv=_inv_chol_d_P0, constant=False)
#             return logp_h + logp_d
        
#         def Mt_logpdf(x_t_m_1, x_t, _params):
#             x_t_m_1 = jnp.atleast_2d(x_t_m_1)
#             x_t = jnp.atleast_2d(x_t)
            
#             h_t_m_1 = x_t_m_1[:, :dh]
#             d_t_m_1 = x_t_m_1[:, dh:]
#             h_t = x_t[:, :dh]
#             d_t = x_t[:, dh:]

#             h_pred = h0 + phi_h * (h_t_m_1 - h0)
#             d_pred = d0 + phi_d * (d_t_m_1 - d0)

#             logp_h = mvn_logpdf(h_t, h_pred, None, chol_inv=_inv_chol_Sigma_h, constant=False)
#             logp_d = mvn_logpdf(d_t, d_pred, None, chol_inv=_inv_chol_Sigma_d, constant=False)
#             return logp_h + logp_d
        
#         Gamma_0 = lambda x: log_potential(x, ys[0], dh) + M0_logpdf(x)
#         Gamma_t = lambda x_t_m_1, x_t, y: log_potential(x_t, y, dh) + Mt_logpdf(x_t_m_1, x_t, None)

#     else:
#         raise NotImplementedError(f"Unknown style: {style}, choose from 'bootstrap'")

#     M0 = M0_rvs, M0_logpdf
#     Mt = Mt_rvs, Mt_logpdf, ys[1:]
#     Gamma_t_plus_params = Gamma_t, ys[1:]

#     kernel = lambda key, state, *_: csmc.kernel(key, state[0], state[1], M0, Gamma_0, Mt, Gamma_t_plus_params, N=N,
#                                                 **kwargs)
#     init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

#     def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
#         samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
#         return samples, flags

#     def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
#                            n_steps, verbose, **_kwargs):
#         return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
#                                         initial_delta, n_steps, verbose, **_kwargs)

#     return kernel, init, adaptation_routine, sampling_routine_fn


# def get_tp_csmc_kernel(ys, m0, P0, Q, phi, N, style="marginal", stop_gradient=False, **kwargs):
#     """
#     Implement the Twisted Particle-mGRAD kernel for the scalable stochastic volatility model.

#     :param ys: Observations (returns)
#     :param m0: (h0, d0) Initial means for the latent states
#     :param P0: (h_P, d_P) Initial covariances for the latent states
#     :param Q: (Sigma_h, Sigma_d) Process noise covariances for the latent states
#     :param phi: (phi_h, phi_d) Persistence coefficients for h and d
#     :param N: Number of particles
#     :param style: Style of CSMC kernel ('bootstrap' supported)
#     :param kwargs: Additional keyword arguments
#     """

#     _check_param_shapes(m0, P0, Q, phi)

#     h0, d0 = m0
#     dh, dd = h0.shape[0], d0.shape[0]
#     phi_h, phi_d = phi

#     Sigma_h, Sigma_d = Q

#     if Sigma_h.ndim == 1:
#         Sigma_h = jnp.diag(Sigma_h)
#     if Sigma_d.ndim == 1:
#         Sigma_d = jnp.diag(Sigma_d)

#     Q = block_diag(Sigma_h, Sigma_d)

#     def as_phi_matrix(phi, d):
#         if jnp.ndim(phi) == 0:          # scalar
#             return phi * jnp.eye(d)
#         if jnp.ndim(phi) == 1:          # per-dimension coefficients
#             return jnp.diag(phi)
#         return phi                      # already a matrix

#     phi_h = as_phi_matrix(phi_h, dh)
#     phi_d = as_phi_matrix(phi_d, dd)

#     h_P0, d_P0 = P0
#     P0 = block_diag(h_P0, d_P0)
#     m0 = jnp.concatenate([h0, d0], axis=0)
#     F = block_diag(phi_h, phi_d)
#     b = jnp.concatenate([h0 - phi_h @ h0, d0 - phi_d @ d0], axis=0)

#     def r0(x):
#         if stop_gradient:
#             return log_potential(jax.lax.stop_gradient(x), ys[0], dh)
#         return log_potential(x, ys[0], dh)

#     def rt(_, x, y):
#         if stop_gradient:
#             return log_potential(jax.lax.stop_gradient(x), y, dh)
#         return log_potential(x, y, dh)

#     rt_plus_params = rt, ys[1:]

#     def mut(x, _):
#         h_t_m_1 = x[:dh]
#         d_t_m_1 = x[dh:]

#         h_pred = h0 + phi_h * (h_t_m_1 - h0)
#         d_pred = d0 + phi_d * (d_t_m_1 - d0)

#         return jnp.concatenate([h_pred, d_pred], axis=0)
    
#     Qs = jnp.repeat(Q[None, ...], ys.shape[0] - 1, axis=0)

#     if style == 'marginal':
#         kernel = tpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
#     elif style == 'filtering':
#         kernel = atpf.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
#     elif style == 'smoothing':
#         kernel = atps.get_kernel(m0, P0, r0, mut, Qs, rt_plus_params, N=N, **kwargs)
#     elif style == 'twisted':
#         Fs = jnp.repeat(F[None, ...], ys.shape[0] - 1, axis=0)
#         bs = jnp.repeat(b[None, ...], ys.shape[0] - 1, axis=0)
#         kernel = t_atpf.get_kernel(m0, P0, r0, Fs, bs, Qs, rt_plus_params, N=N, **kwargs)
#     else:
#         raise NotImplementedError(
#             f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing', 'twisted'")

#     wrapped_kernel = lambda key, state, delta: kernel(key, state[0], state[1], delta, delta)

#     init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

#     def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
#         samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
#         return samples, flags

#     def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
#                            n_steps, verbose, **kwargs_):
#         if style == "twisted":
#             return t_atpf.delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
#                                                    initial_delta, n_steps, verbose, **kwargs_)
#         return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
#                                         initial_delta, n_steps, verbose, **kwargs_)

#     return wrapped_kernel, init, adaptation_routine, sampling_routine_fn


# def get_mala_csmc_kernel(ys, m0, P0, Q, phi, N, style="marginal", **kwargs):
#     """
#     Implement the Particle-aMALA algorithm for the scalable stochastic volatility model.
    
#     :param ys: Description
#     :param m0: Description
#     :param P0: Description
#     :param Q: Description
#     :param phi: Description
#     :param N: Description
#     :param style: Description
#     :param kwargs: Description
#     """

#     _check_param_shapes(m0, P0, Q, phi)

#     h0, d0 = m0
#     dh, dd = h0.shape[0], d0.shape[0]
#     phi_h, phi_d = phi
#     Sigma_h, Sigma_d = Q

#     if Sigma_h.ndim == 1:
#         Sigma_h = jnp.diag(Sigma_h)
#     if Sigma_d.ndim == 1:
#         Sigma_d = jnp.diag(Sigma_d)

#     h_P0, d_P0 = P0

#     chol_h_P0 = jnp.linalg.cholesky(h_P0)
#     chol_d_P0 = jnp.linalg.cholesky(d_P0)
#     chol_Sigma_h = jnp.linalg.cholesky(Sigma_h)
#     chol_Sigma_d = jnp.linalg.cholesky(Sigma_d)

#     _inv_chol_h_P0 = solve_triangular(chol_h_P0, jnp.eye(h0.shape[0]), lower=True)
#     _inv_chol_d_P0 = solve_triangular(chol_d_P0, jnp.eye(d0.shape[0]), lower=True)
#     _inv_chol_Sigma_h = solve_triangular(chol_Sigma_h, jnp.eye(h0.shape[0]), lower=True)
#     _inv_chol_Sigma_d = solve_triangular(chol_Sigma_d, jnp.eye(d0.shape[0]), lower=True)

#     @partial(jnp.vectorize, signature='(d)->()')
#     def Gamma_0(x):
#         x = jnp.atleast_2d(x)

#         h = x[:, :dh]
#         d = x[:, dh:]

#         out = log_potential(x, ys[0], dh)
#         log_pdf_h = mvn_logpdf(h, h0, None, chol_inv=_inv_chol_h_P0, constant=False)
#         log_pdf_d = mvn_logpdf(d, d0, None, chol_inv=_inv_chol_d_P0, constant=False)
#         return out[0] + jnp.concatenate([log_pdf_h, log_pdf_d], axis=0).sum()

#     @partial(jnp.vectorize, signature='(d),(d),(k)->()')
#     def Gamma_t(x_t_m_1, x_t, y):
#         x_t_m_1 = jnp.atleast_2d(x_t_m_1)
#         x_t = jnp.atleast_2d(x_t)
        
#         h_t_m_1 = x_t_m_1[:, :dh]
#         d_t_m_1 = x_t_m_1[:, dh:]
#         h_t = x_t[:, :dh]
#         d_t = x_t[:, dh:]

#         h_pred = h0 + phi_h * (h_t_m_1 - h0)
#         d_pred = d0 + phi_d * (d_t_m_1 - d0)

#         out = log_potential(x_t, y, dh)
#         log_pdf_h = mvn_logpdf(h_t, h_pred, None, chol_inv=_inv_chol_Sigma_h, constant=False)
#         log_pdf_d = mvn_logpdf(d_t, d_pred, None, chol_inv=_inv_chol_Sigma_d, constant=False)
#         return out[0] + jnp.concatenate([log_pdf_h, log_pdf_d], axis=0).sum()

#     Gamma_t_plus_params = Gamma_t, ys[1:]

#     if style == "filtering":
#         kernel = lambda key, state, delta: alf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
#                                                       delta, N=N, **kwargs)
#     elif style == "smoothing":
#         kernel = lambda key, state, delta: als.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
#                                                       delta, N=N, **kwargs)
#     elif style == "marginal":
#         kernel = lambda key, state, delta: lf.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params, delta,
#                                                      delta, N=N, **kwargs)
#     else:
#         raise NotImplementedError(f"Unknown style: {style}, choose from 'marginal', 'filtering', 'smoothing'")
#     init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

#     def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
#         samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
#         return samples, flags

#     def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
#                            n_steps, verbose, **_kwargs):
#         return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
#                                         initial_delta, n_steps, verbose, **_kwargs)

#     return kernel, init, adaptation_routine, sampling_routine_fn


# def get_rw_csmc_kernel(ys, m0, P0, Q, phi, N, **kwargs):
#     """
#     Implement the random-walk CSMC kernel for the scalable stochastic volatility model.

#     :param ys: Observations (returns)
#     :param m0: (h0, d0) Initial means for the latent states
#     :param P0: (P_h, P_d) Initial covariances for the latent states
#     :param sigma0: (Sigma_h, Sigma_d) Process noise standard deviations
#     :param phi: (phi_h, phi_d) Persistence coefficients for h and d
#     :param kwargs: Additional keyword arguments
#     """

#     _check_param_shapes(m0, P0, Q, phi)

#     kwargs.pop("style")

#     h0, d0 = m0
#     dh = h0.shape[0]
#     phi_h, phi_d = phi
#     Sigma_h, Sigma_d = Q

#     if Sigma_h.ndim == 1:
#         Sigma_h = jnp.diag(Sigma_h)
#     if Sigma_d.ndim == 1:
#         Sigma_d = jnp.diag(Sigma_d)

#     h_P0, d_P0 = P0

#     chol_h_P0 = jnp.linalg.cholesky(h_P0)
#     chol_d_P0 = jnp.linalg.cholesky(d_P0)
#     chol_Sigma_h = jnp.linalg.cholesky(Sigma_h)
#     chol_Sigma_d = jnp.linalg.cholesky(Sigma_d)

#     inv_chol_h_P0 = solve_triangular(chol_h_P0, jnp.eye(h0.shape[0]), lower=True)
#     inv_chol_d_P0 = solve_triangular(chol_d_P0, jnp.eye(d0.shape[0]), lower=True)
#     inv_chol_Sigma_h = solve_triangular(chol_Sigma_h, jnp.eye(h0.shape[0]), lower=True)
#     inv_chol_Sigma_d = solve_triangular(chol_Sigma_d, jnp.eye(d0.shape[0]), lower=True)

#     @partial(jnp.vectorize, signature='(d)->()')
#     def Gamma_0(x):
#         x = jnp.atleast_2d(x)

#         h = x[:, :dh]
#         d = x[:, dh:]

#         out = log_potential(x, ys[0], dh)
#         log_pdf_h = mvn_logpdf(h, h0, None, chol_inv=inv_chol_h_P0, constant=False)
#         log_pdf_d = mvn_logpdf(d, d0, None, chol_inv=inv_chol_d_P0, constant=False)
#         return out[0] + jnp.concatenate([log_pdf_h, log_pdf_d], axis=0).sum()

#     @partial(jnp.vectorize, signature='(d),(d),(k)->()')
#     def Gamma_t(x_t_m_1, x_t, y):
#         x_t_m_1 = jnp.atleast_2d(x_t_m_1)
#         x_t = jnp.atleast_2d(x_t)
        
#         h_t_m_1 = x_t_m_1[:, :dh]
#         d_t_m_1 = x_t_m_1[:, dh:]
#         h_t = x_t[:, :dh]
#         d_t = x_t[:, dh:]

#         h_pred = h0 + phi_h * (h_t_m_1 - h0)
#         d_pred = d0 + phi_d * (d_t_m_1 - d0)

#         out = log_potential(x_t, y, dh)
#         log_pdf_h = mvn_logpdf(h_t, h_pred, None, chol_inv=inv_chol_Sigma_h, constant=False)
#         log_pdf_d = mvn_logpdf(d_t, d_pred, None, chol_inv=inv_chol_Sigma_d, constant=False)
#         return out[0] + jnp.concatenate([log_pdf_h, log_pdf_d], axis=0).sum()

#     Gamma_t_plus_params = Gamma_t, ys[1:]

#     kernel = lambda key, state, delta: rw.kernel(key, state[0], state[1], Gamma_0, Gamma_t_plus_params,
#                                                  delta, N=N, **kwargs)

#     init = lambda x: (x, jnp.zeros((x.shape[0],), dtype=int))

#     def sampling_routine_fn(key, state, kernel_, n_steps, verbose, get_samples):
#         samples, flags = sampling_routine(key, state[0], state[1], kernel_, n_steps, verbose, get_samples)
#         return samples, flags

#     def adaptation_routine(key, state, kernel_, target_acceptance, initial_delta,
#                            n_steps, verbose, **_kwargs):
#         return delta_adaptation_routine(key, state[0], state[1], kernel_, target_acceptance,
#                                         initial_delta, n_steps, verbose, **_kwargs)

#     return kernel, init, adaptation_routine, sampling_routine_fn

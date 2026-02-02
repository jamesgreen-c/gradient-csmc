"""
Implements the Particle-aGrad kernel of the paper.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp

from gradient_csmc.csmc import kernel as csmc_kernel
from gradient_csmc.utils.diag_mvn import diag_mvn_logpdf

_STABLE_ZERO = 1e-15


def get_kernel(mu0: Array, P0_DIAG: Array, r0: Callable,
               mut: Union[Callable, tuple[Callable, Any]], SIGMAs,
               rt: Union[Callable, tuple[Callable, Any]],
               resampling_func: Callable, ancestor_move_func: Callable, N: int,
               backward: bool = False
               ):
    """
    Constructor for the Particle-aGrad kernel.

    Parameters
    ----------
    mu0:
        Initial distribution mean vector.
    P0:
        Initial distribution diagonal standard deviations.
    r0:
        Initial distribution log-density.
    mut:
        Transition function. Either a callable or a tuple of a callable and its parameters.
    Qs:
        Array of diagonal standard deviations vectors. The first element is the stdev at time 0: p(x0) = N(m0, Qs[0]).
        The rest is the stdev vector of transition functions: p(xt | xt-1) = N(xt | µ(xt-1), Qs[t]).
    rt:
        Potential function. Either a callable or a tuple of a callable and its parameters.
    resampling_func:
        Resampling function.
    ancestor_move_func:
        Ancestor move function.
    N:
        Number of particles.
    backward:
        Whether to run the backward sampling kernel.

    Returns
    -------
    kernel: Callable
        ATP-CSMC kernel function.

    """

    Qs_ = jnp.insert(SIGMAs, 0, P0_DIAG, axis=0)**2    # element wise square for variances
    get_proposal_params = make_proposal(Qs_)

    mut, mut_params = mut if isinstance(mut, tuple) else (mut, None)
    rt, rt_params = rt if isinstance(rt, tuple) else (rt, None)

    val_and_grad_r0 = jnp.vectorize(jax.value_and_grad(r0), signature='(d)->(),(d)')
    val_and_grad_rt = jnp.vectorize(jax.value_and_grad(rt, argnums=1), signature='(d),(d)->(),(d)', excluded=(2,))

    def kernel(key: PRNGKey, x_star: Array, b_star: Array, ells: Array, deltas: Array):
        """
        Implements the Particle-aGrad kernel of the paper.

        Parameters
        ----------
        key: PRNGKey
            JAX PRNGKey.
        x_star:
            Reference trajectory.
        b_star:
            Reference ancestor indices.
        ells:
            Proposal variance scaling.
        deltas:
            Gradient step-size.

        Returns
        -------
        x_star: Array
            New reference trajectory.
        b_star: Array
            New reference ancestor indices.

        """
        ###############################
        #        HOUSEKEEPING         #
        ###############################

        Ks, chols_prop = get_proposal_params(ells)

        T, d_x = x_star.shape
        key_csmc, _, key_aux = jax.random.split(key, 3)
        aux_std_devs = jnp.sqrt(0.5 * ells)

        _, grad_log_w_star_0 = val_and_grad_r0(x_star[0])
        _, grad_log_w_star = jax.vmap(val_and_grad_rt)(x_star[:-1], x_star[1:], rt_params)
        grad_log_w_star = jnp.insert(grad_log_w_star, 0, grad_log_w_star_0, axis=0)

        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        aux_vars = x_star + 0.5 * deltas[:, None] * grad_log_w_star + aux_std_devs[:, None] * eps_aux

        ###############################################
        #       Proposal and weight functions         #
        ###############################################

        spec_M0_rvs = partial(M0_rvs, params=(aux_vars[0], mu0, Ks[0], chols_prop[0]))
        spec_M0_logpdf = partial(M0_logpdf, params=(aux_vars[0], mu0, Ks[0], chols_prop[0]))
        spec_Mt_rvs = partial(Mt_rvs, mut=mut)
        spec_Mt_logpdf = partial(Mt_logpdf, mut=mut)
        spec_Gamma_0 = partial(Gamma_0, val_and_grad_r0=val_and_grad_r0, params=(aux_vars[0], mu0, P0_DIAG,
                                                                                 deltas[0], ells[0]))
        spec_Gamma_t = partial(Gamma_t, val_and_grad_rt=val_and_grad_rt, mut=mut)

        Mt_params = (aux_vars[1:], Ks[1:], chols_prop[1:]), mut_params
        Mt = spec_Mt_rvs, spec_Mt_logpdf, Mt_params
        Gamma_t_params = aux_vars[1:], deltas[1:], ells[1:], SIGMAs, mut_params, rt_params
        Gamma_t_plus_params = spec_Gamma_t, Gamma_t_params
        M0 = spec_M0_rvs, spec_M0_logpdf

        ###########################
        #       Call CSMC         #
        ###########################
        return csmc_kernel(key_csmc, x_star, b_star, M0, spec_Gamma_0, Mt, Gamma_t_plus_params, resampling_func,
                           ancestor_move_func, N, backward)

    return kernel


# def make_proposal(Gammas, get_K_inv=False):

#     def _get_proposal_params(ell, Gamma):
#         """
#         Gamma is a diagonal vector of standard deviations        
#         """
#         inv_pred_cov = 1. / (Gamma + ell / 2)
#         K = Gamma[:, None] * inv_pred_cov

#         temp = jnp.clip(Gamma - Gamma ** 2 * inv_pred_cov, _STABLE_ZERO, jnp.inf)
#         temp = jnp.sqrt(temp)
#         chol = temp[None, :]
#         chol_inv = 1. / chol
#         if get_K_inv:
#             K_inv = 1. / K
#             return K, chol, chol_inv, K_inv
#         return K, chol, chol_inv

#     def get_proposal_params(ells):
#         return jax.vmap(_get_proposal_params)(ells, Gammas)

#     return get_proposal_params

def make_proposal(q_diag, get_K_inv=False):
    """
    q_diag: (..., D) diagonal of covariance Q (i.e. variances), NOT std devs.
    """

    def _get_proposal_params(ell, q):
        inv_pred_cov = 1.0 / (q + ell / 2.0)          # (D,)
        K = q * inv_pred_cov                          # (D,)  = diag of gain

        c_diag = jnp.clip(q - q**2 * inv_pred_cov, _STABLE_ZERO, jnp.inf)  # (D,)
        chol = jnp.sqrt(c_diag)                       # (D,)

        # chol = jnp.clip(chol, _STABLE_ZERO, jnp.inf)
        # chol_inv = 1.0 / chol                         # (D,)

        if get_K_inv:
            K_inv = 1.0 / jnp.clip(K, _STABLE_ZERO, jnp.inf)     # diag: (γ+ell/2)/γ ; beware γ=0 -> inf
            return K, chol, K_inv                                # return K, chol, chol_inv, K_inv
        
        return K, chol #, chol_inv

    def get_proposal_params(ells):
        # Case A: ells has shape (T,) and q_diag has shape (T, D): pairwise by t
        return jax.vmap(_get_proposal_params)(ells, q_diag)

        # Case B (common): one ell scalar applied to all q's:
        # return jax.vmap(lambda q: _get_proposal_params(ells, q))(q_diag)
    return get_proposal_params


def M0_rvs(key, N, params):
    u, mu0, K, chol_prop = params
    dx = mu0.shape[0]
    mean_0 = mu0 + K * (u - mu0)
    eps = jax.random.normal(key, (N, dx))
    # jax.debug.print("mean_0: {}", mean_0)
    # jax.debug.print("P0: {}", chol_prop @ chol_prop.T)
    return mean_0[None, :] + eps * chol_prop


def M0_logpdf(x, params):
    u, mu0, K, chol_prop = params
    mean_0 = mu0 + K * (u - mu0)
    return diag_mvn_logpdf(x, mean_0, chol_prop, constant=False)


def Mt_rvs(key, x_prev, params, mut):
    (u, K, chol_prop), mu_params = params
    N_, dx = x_prev.shape
    mean_prior = mut(x_prev, mu_params)
    mean_t = mean_prior + K * (u[None, :] - mean_prior)
    eps = jax.random.normal(key, (N_, dx))
    return mean_t + eps * chol_prop


def Mt_logpdf(x_prev, x, params, mut):
    (u, K, chol_prop), mu_params = params
    mean_prior = mut(x_prev, mu_params)
    mean_t = mean_prior + K * (u[None, :] - mean_prior)
    return diag_mvn_logpdf(x, mean_t, chol_prop, constant=False)


def Gamma_0(x, val_and_grad_r0, params):
    u, mu0, P0_DIAG, delta, ell = params
    out = diag_mvn_logpdf(x, mu0, P0_DIAG, constant=False)
    r0_val, r0_grad = val_and_grad_r0(x)
    out += r0_val
    correction_term = -jnp.sum((u - x - 0.5 * delta * r0_grad) ** 2, -1) / ell
    out += correction_term
    return out


def Gamma_t(x_prev, x, params, val_and_grad_rt, mut):
    u, delta, ell, sigma, mu_params, rt_params_here = params
    out = diag_mvn_logpdf(x, mut(x_prev, mu_params), sigma, constant=False)

    rt_val, rt_grad = val_and_grad_rt(x_prev, x, rt_params_here)
    out += rt_val
    correction_term = -jnp.sum((u - x - 0.5 * delta * rt_grad) ** 2, -1) / ell
    out += correction_term
    return out

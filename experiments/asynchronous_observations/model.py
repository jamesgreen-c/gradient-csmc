from functools import partial

import jax
from jax import numpy as jnp
from jax.scipy.stats import norm


"""
1. Return to a linear Gaussian get_data function
2. Sample T scalar indices for the corresponding observation y_t(i)
3. Change log potential to take index as input and return a Guassian evaluation with mean equal to x_ti
"""


@partial(jax.jit, static_argnums=(2, 3))
def get_data(key, sigma, dim, T):
    x_key, y_key, inds_key = jax.random.split(key, 3)

    x0 = sigma * jax.random.normal(x_key, (dim,))
    eps_xs = jax.random.normal(x_key, (T, dim,))
    eps_ys = jax.random.normal(y_key, (T,))
    
    inds = jax.random.choice(inds_key, dim, (T,))

    def body(x_k, inps):
        eps_x, eps_y, ind_k = inps
        y_k = x_k[ind_k] + eps_y   # y_k is a 1D Gaussian RV with mean x_k[i] and std 1
        x_kp1 = x_k + sigma * eps_x
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys, inds))
    return xs, ys, inds


@partial(jnp.vectorize, signature="(n),(),()->()")
def log_potential(x, y, ind):
    val = norm.logpdf(y, x[ind])
    return jnp.sum(val)


def log_likelihood(x, y, ind):
    return jnp.sum(log_potential(x, y, ind))


def log_pdf(xs, ys, inds, sigma):
    
    # extract the corresponding dimensions
    xs = xs[jnp.arange(xs.shape[0]), inds]      # (T,)
    xs = xs[:, None]                            # (T, 1)
    
    def _logpdf(zs):
        out = jnp.sum(norm.logpdf(zs[0], scale=sigma))
        out += jnp.sum(norm.logpdf(zs[1:], zs[:-1], sigma))
        out += jnp.sum(norm.logpdf(zs, ys))
        return out

    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)

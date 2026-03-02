from functools import partial

import jax
from jax import numpy as jnp

from jax.scipy.stats import norm


"""
1. Return to a linear Gaussian get_data function
2. Sample T scalar indices for the corresponding observation y_t(i)
3. Change log potential to take index as input and return a Guassian evaluation with mean equal to x_ti
"""

"""
USE A CONTINUOUS TIME MODEL WITH THE GAUSSIAN TRANSITION DYNAMICS (THIS IS WHAT
    ALLOWS US TO CONDENSE OBSERVATIONS INTO A SMALL AMOUNT OF TIME IE [0, 1])
THEREFORE NUMBER OF STEPS WILL BE CONTROLLED WITH DIMENSION (NO NEED FOR T)
RUN EXPERIMENTS WITH INCREASING D
IF DEGENERACY OCCURS, RUN EXPERIMENTS WITH INCREASING PARTICLES N TO SEE HOW
    MANY ARE NEEDED TO SOLVE IT
SEE IF RAO-BLACKWELLISATION SOLVES THE ISSUE
"""


def random_corr_chol(key, dim, jitter=1e-6):
    # SPD -> correlation -> Cholesky
    M = jax.random.normal(key, (dim, dim))
    C = M @ M.T + jitter * jnp.eye(dim)      # SPD
    d = jnp.sqrt(jnp.diag(C))
    R = C / (d[:, None] * d[None, :])        # correlation (diag=1)
    L = jnp.linalg.cholesky(R)
    return L, R


@partial(jax.jit, static_argnums=(2, 3))
def get_data(key, sigma, dim, phi: float = 1.0):
    dt = 1.0 / dim
    T = dim

    key_corr, key_x0, key_zx, key_y = jax.random.split(key, 4)

    # R = L L^T (correlation)
    L, R = random_corr_chol(key_corr, dim)

    # x0 ~ N(0, sigma^2 R):
    chol_P0 = sigma * L
    m0 = jnp.zeros((dim, ))
    z0 = jax.random.normal(key_x0, (dim,))
    x0 = chol_P0 @ z0

    # OU discretisation
    A = jnp.exp(-phi * dt)
    sigma_dt = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * phi * dt)) / (2.0 * phi))
    # sigma_dt = sigma * jnp.sqrt((jnp.exp(2.0 * phi * dt) - 1.0) / (2.0 * phi))
    chol_Q = sigma_dt * L

    # standard normals for the step noise
    eps_xs = jax.random.normal(key_zx, (T, dim))      # each row ~ N(0, I)
    eps_ys = jax.random.normal(key_y, (T, dim))           # scalar obs noise

    def body(x_k, inps):
        eps_x, eps_y = inps
        y_k = x_k + eps_y

        # correlated OU increment
        eta_k = chol_Q @ eps_x                    # (dim,)
        x_kp1 = A * x_k + eta_k
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys))
    return xs, m0, ys, chol_P0, chol_Q


@partial(jnp.vectorize, signature="(n),(n)->()")
def log_potential(x, y):
    val = norm.logpdf(y, x)
    return jnp.sum(val)


def log_likelihood(x, y):
    return jnp.sum(log_potential(x, y))


def log_pdf(xs, ys, sigma):    
    def _logpdf(zs):
        out = jnp.sum(norm.logpdf(zs[0], scale=sigma))
        out += jnp.sum(norm.logpdf(zs[1:], zs[:-1], sigma))
        out += jnp.sum(norm.logpdf(zs, ys))
        return out
    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)

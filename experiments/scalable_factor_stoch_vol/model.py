"""

Implementation of the scalable factor stochastic volatility model from Tistias 2023.
While their model is generalisable to varying parameters over time, our implmentation
here focuses on the constant-parameter case for simplicity. That is to say, we have:
    1. phi_ijt = phi_ij
    2. phi_it = phi_i
    3. omega_ijt = omega_ij
    4. sigma_ijt = sigma_ij
    5. sigma_it = sigma_i 

In this model our latent states x_t is actually a combination of two components:
    1. h_t: the log-eigenvalues of the covariance matrix at time t
    2. delta_t: the log-transformed rotation angles for Givens matrices making up the 
       eigenvectors at time t
We therefore have K*(K-1) + K latent state values at each time t for a 
K-dimensional underlying factor of the D-dimensional observation y_t. 
We therefore store the latent state as a vector of length K*(K-1) + K.

We model the observations y_t as being generated from a factor model:
    y_t = B f_t + V_t

"""

from functools import partial

import jax
import jax.random as jr
import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular, block_diag
from jax.scipy.stats import norm

from gradient_csmc.utils.math import mvn_logpdf
from experiments.scalable_stochastic_volatility.utils import _check_param_shapes

def as_square(A):
    return jnp.diag(A) if A.ndim == 1 else A

@partial(jax.jit, static_argnums=(0, 1,))
def get_dynamics(D: int, K: int = None):
    # TODO: make these parameters inputs to the function and experiment script 
    
    K = D if K is None else K

    h0 = jnp.zeros((K,))
    d0 = jnp.zeros((K*(K-1) // 2,))
    m0 = jnp.concatenate([h0, d0], axis=0)

    phi_h = as_square(0.8 * jnp.linspace(1, 0.1, K))
    phi_d = as_square(0.7 * jnp.linspace(1, 0.1, K*(K-1) // 2))   # jnp.ones((K*(K-1) // 2,)))
    F = block_diag(phi_h, phi_d)
    b = jnp.concatenate([h0 - phi_h @ h0, d0 - phi_d @ d0], axis=0)

    # phi_h = as_square(0.95 * jnp.linspace(1, 0, K))
    # phi_d = as_square(0.90 * jnp.linspace(1, 0, K*(K-1) // 2))   # jnp.ones((K*(K-1) // 2,)))
    # F = block_diag(phi_h, phi_d)
    # b = jnp.concatenate([h0 - phi_h @ h0, d0 - phi_d @ d0], axis=0)

    sigma_h = jnp.diag(0.3 * jnp.ones((K,)))
    sigma_d = jnp.diag(0.3 * jnp.ones((K*(K-1) // 2,)))

    # sigma_h = jnp.diag(0.1 * jnp.ones((K,)))
    # sigma_d = jnp.diag(0.1 * jnp.ones((K*(K-1) // 2,)))

    h_P0 = (sigma_h**2) / (1 - phi_h**2)
    d_P0 = (sigma_d**2) / (1 - phi_d**2)
    P0 = block_diag(h_P0, d_P0)

    if sigma_h.ndim == 1:
        sigma_h = jnp.diag(sigma_h)
    if sigma_d.ndim == 1:
        sigma_d = jnp.diag(sigma_d)

    Q = block_diag(sigma_h**2, sigma_d**2)

    B = jnp.eye(D)[:, :K] # D x K
    V = jnp.diag(0.1 * jnp.ones((D,)))  # D x D

    return m0, P0, Q, F, b, B, V


@partial(jax.jit, static_argnums=(1,))
def make_single_givens_matrix(omega, D: int, i, j):
    """
    Make a single Givens rotation matrix of dimension D x D that rotates in the (i, j) plane by angle omega

    :param theta: The rotation angle
    :param D: The dimension of the square Givens matrix
    :param i: The first index of the plane to rotate in
    :param j: The second index of the plane to rotate in
    :return: A D x D Givens rotation matrix
    """
    G = jnp.eye(D)
    c = jnp.cos(omega)
    s = jnp.sin(omega)

    G = G.at[i, i].set(c)
    G = G.at[j, j].set(c)
    G = G.at[i, j].set(-s)
    G = G.at[j, i].set(s)

    return G


@partial(jax.jit, static_argnums=(1,))
def make_givens_matrices(omegas, K: int):
    """
    omegas is a K*(K-1) vector. Reshape to K x K matrix and make Givens matrices

    :param omegas: (K*(K-1),) containing the rotation angles for Givens matrices
    :param K: The dimension K of the factor space
    """

    # i<j pairs
    I, J = jnp.triu_indices(K, k=1)  # shape (K,), (K,)

    G = jax.vmap(
        lambda omega, i, j: make_single_givens_matrix(omega, K, i, j), 
        in_axes=(0, 0, 0)
    )(omegas, I, J)  # (K, D, D)
    return G


@partial(jax.jit, static_argnums=(1,))
def make_covariance_matrix(x, K: int):
    """
    Construct the covariance matrix from:

    S = P Lambda P^T
    P = prod_{i<j} G_ij(omega_ij)
    h_i = log(lambda_i)
    d_ij = log( (pi/2 - omega_ij) / (pi/2 + omega_ij) )
    
    :param x: Latent state vector of shape (D*(D-1) + D,) containing h and d
    """

    h = x[:K]
    d = x[K:]

    # make eigenvalues
    lambdas = jnp.exp(h)

    # make omegas and eigenvectors
    omegas = 0.5 * jnp.pi * (1 - jnp.exp(d)) / (1 + jnp.exp(d))
    I, J = jnp.triu_indices(K, k=1)  # shape (K,), (K,)
    # G = make_givens_matrices(omegas, D)  # (K, D, D)

    def apply_givens_rotations(P, inputs):
        omega, i, j = inputs
        c = jnp.cos(omega)
        s = jnp.sin(omega)

        # Right-multiply by G_ij: only columns i and j change.
        Pi = P[:, i]
        Pj = P[:, j]
        P = P.at[:, i].set(c * Pi + s * Pj)
        P = P.at[:, j].set(-s * Pi + c * Pj)
        return P, None

    P0 = jnp.eye(K)
    P, _ = jax.lax.scan(apply_givens_rotations, P0, (omegas, I, J))  # (D, D)

    Sigma = P @ (lambdas[:, None] * P.T)  # avoids explicit diag(lambdas)
    return Sigma
    # G = make_givens_matrices(omegas, K)  # (B, K, K)

    # def matmul_scan(carry, A):
    #     return carry @ A, None
    # P, _ = jax.lax.scan(matmul_scan, jnp.eye(K), G)  # (K, K)
    
    # # make covariance matrix and sample y_k
    # Sigma = P @ jnp.diag(lambdas) @ P.T
    # return Sigma


@partial(jax.jit, static_argnums=(8, 9))
def get_data(key, m0, P0, Q, F, b, B, V, K:int, T: int):
    
    init_key, sampling_key = jr.split(key, 2)

    inv_chol_P0 = jnp.linalg.cholesky(jnp.linalg.inv(P0))
    inv_chol_Q = jnp.linalg.cholesky(jnp.linalg.inv(Q))

    x1 = jr.multivariate_normal(init_key, m0, P0)

    def body(x_k, key_k):
        state_key, factor_key, observation_key = jr.split(key_k, 3)
        
        Sigma_k = make_covariance_matrix(x_k, K)
        
        # sample factor f_k
        f_k = jr.multivariate_normal(
            factor_key,
            mean=jnp.zeros((K,)),
            cov=Sigma_k
        )

        # sample observation y_k
        y_k = jr.multivariate_normal(
            observation_key,
            mean=B @ f_k,
            cov=V
        )

        # sample next state x_k+1
        x_kp1 = F @ x_k + b + jr.multivariate_normal(
            state_key,
            mean=jnp.zeros(x_k.shape),
            cov=Q
        )
        return x_kp1, (x_k, f_k, y_k)

    _, (xs, fs, ys) = jax.lax.scan(body, x1, jax.random.split(sampling_key, T))
    return xs, fs, ys, inv_chol_P0, inv_chol_Q


@jax.jit
def log_potential(x, y, B, V):
    """ 
    Rao Blackwellised log potential for the scalable factor stochastic volatility model
    """
    K = B.shape[1]

    @partial(jnp.vectorize, signature="(k),(n)->()")
    def _log_potential(x, y):
        Sigma_f = make_covariance_matrix(x, K)
        Sigma_y = B @ Sigma_f @ B.T + V @ jnp.eye(V.shape[0])
        Sigma_y = Sigma_y + 1e-6 * jnp.eye(Sigma_y.shape[0])
        # val = norm.logpdf(y, scale=Sigma_y)
        
        chol_Sigma_y = jnp.linalg.cholesky(Sigma_y)
        _chol_Sigma_y_inv = solve_triangular(chol_Sigma_y, jnp.eye(Sigma_y.shape[0]), lower=True)
        val = mvn_logpdf(y, jnp.zeros(Sigma_y.shape[0]), None, _chol_Sigma_y_inv, constant=False)

        return jnp.nansum(val)  # in case the scale is infinite, we get nan, but we want 0
    return _log_potential(x, y)


def log_likelihood(x, y, B, V):
    return jnp.sum(log_potential(x, y, B, V))


def log_pdf(xs, ys, m0, inv_chol_P0, F, b, inv_chol_Q, B, V):
    def _logpdf(zs):
        out = mvn_logpdf(zs[0], m0, None, inv_chol_P0, constant=False)
        # pred_xs = F @ zs[:-1] + b
        # pred_xs = jnp.einsum('...j,ij,i->...i', zs[:-1], F, b)
        pred_xs = jnp.einsum('ij,...j->...i', F, zs[:-1]) + b
        out += jnp.sum(mvn_logpdf(zs[1:], pred_xs, None, inv_chol_Q, constant=False))
        out += log_likelihood(zs, ys, B, V)
        return out
    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    T = 300
    D = 10
    K = 3

    if D < 2:
        K = 1

    m0, P0, Q, F, b, B, V = get_dynamics(D, K)

    print(m0.shape)
    print(P0.shape)
    print(Q.shape)
    print(F.shape)
    print(b.shape)
    print(B.shape)
    print(V.shape)

    xs, fs, ys, inv_chol_P0, inv_chol_Q = get_data(key, m0, P0, Q, F, b, B, V, K, T)
    print(xs.shape, fs.shape, ys.shape)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(D, 2, figsize=(15, 5*D))
    if D > 1:
        zs = (B @ fs.T).T # (T, D)
        print(zs.shape)
        for d in range(D):
            ax[d, 0].plot(zs[:, d])
            ax[d, 0].set_title(f"Latent state h dimension {d}")
            ax[d, 1].plot(ys[:, d])
            ax[d, 1].set_title(f"Observations y dimension {d}")
    else:
        ax[0].plot(xs[:, 0])
        ax[0].set_title(f"Latent state h")
        ax[1].plot(ys[:, 0])
        ax[1].set_title(f"Observations y")
    
    plt.tight_layout()
    plt.savefig("experiments/scalable_factor_stoch_vol/plots/data.png")
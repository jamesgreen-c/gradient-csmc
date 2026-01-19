"""

Implementation of the scalable stochastic volatility model from Tistias 2023.
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
We therefore have D*(D-1) + D latent state values at each time t for a 
D-dimensional observation y_t. We therefore store the latent state as a 
vector of length D*(D-1) + D
"""

from functools import partial

import jax
import jax.random as jr
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular, block_diag

from gradient_csmc.utils.math import mvn_logpdf


def as_square(A):
    return jnp.diag(A) if A.ndim == 1 else A


@partial(jax.jit, static_argnums=(0,))
def get_dynamics(D: int):
    # TODO: make these parameters inputs to the function and experiment script

    h0 = jnp.zeros((D,))
    d0 = jnp.zeros((D*(D-1) // 2,))
    m0 = jnp.concatenate([h0, d0], axis=0)

    phi_h = as_square(0.8 * jnp.linspace(1, 0.1, D))
    phi_d = as_square(0.7 * jnp.linspace(1, 0.1, D*(D-1) // 2))
    F = block_diag(phi_h, phi_d)
    b = jnp.concatenate([h0 - phi_h @ h0, d0 - phi_d @ d0], axis=0)

    sigma_h = jnp.diag(0.3 * jnp.ones((D,)))
    sigma_d = jnp.diag(0.3 * jnp.ones((D*(D-1) // 2,)))

    h_P0 = (sigma_h**2) / (1 - phi_h**2)
    d_P0 = (sigma_d**2) / (1 - phi_d**2)
    P0 = block_diag(h_P0, d_P0)
    
    if sigma_h.ndim == 1:
        sigma_h = jnp.diag(sigma_h)
    if sigma_d.ndim == 1:
        sigma_d = jnp.diag(sigma_d)

    Q = block_diag(sigma_h**2, sigma_d**2)
    
    return m0, P0, Q, F, b


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
def make_givens_matrices(omegas, D: int):
    """
    omegas is a D*(D-1) vector. Reshape to D x D matrix and make Givens matrices

    :param omegas: (D*(D-1),) containing the rotation angles for Givens matrices
    :param D: The dimension D of the observation space
    """

    # i<j pairs, length K = D*(D-1)//2
    I, J = jnp.triu_indices(D, k=1)  # shape (K,), (K,)

    G = jax.vmap(
        lambda omega, i, j: make_single_givens_matrix(omega, D, i, j), 
        in_axes=(0, 0, 0)
    )(omegas, I, J)  # (K, D, D)
    return G


@partial(jax.jit, static_argnums=(1,))
def make_covariance_matrix(x, D: int):
    """
    Construct the covariance matrix from:

    S = P Lambda P^T
    P = prod_{i<j} G_ij(omega_ij)
    h_i = log(lambda_i)
    d_ij = log( (pi/2 - omega_ij) / (pi/2 + omega_ij) )
    
    :param x: Latent state vector of shape (D*(D-1) + D,) containing h and d
    """

    h = x[:D]
    d = x[D:]

    # make eigenvalues
    lambdas = jnp.exp(h)

    # make omegas and eigenvectors
    omegas = 0.5 * jnp.pi * (1 - jnp.exp(d)) / (1 + jnp.exp(d))
    G = make_givens_matrices(omegas, D)  # (K, D, D)

    def matmul_scan(carry, A):
        return carry @ A, None
    P, _ = jax.lax.scan(matmul_scan, jnp.eye(D), G)  # (D, D)
    
    # make covariance matrix and sample y_k
    Sigma = P @ jnp.diag(lambdas) @ P.T
    return Sigma


@partial(jax.jit, static_argnums=(6, 7))
def get_data(key, m0, P0, Q, F, b, D: int, T: int):
    
    init_key, sampling_key = jr.split(key, 2)

    inv_chol_P0 = jnp.linalg.cholesky(jnp.linalg.inv(P0))
    inv_chol_Q = jnp.linalg.cholesky(jnp.linalg.inv(Q))

    x1 = jr.multivariate_normal(init_key, m0, P0)

    def body(x_k, key_k):
        state_key, observation_key = jr.split(key_k, 2)
        
        Sigma_k = make_covariance_matrix(x_k, D)
        
        # sample observation y_k
        y_k = jr.multivariate_normal(
            observation_key,
            mean=jnp.zeros((D,)),
            cov=Sigma_k
        )

        # sample next state x_k+1
        x_kp1 = F @ x_k + b + jr.multivariate_normal(
            state_key,
            mean=jnp.zeros(x_k.shape),
            cov=Q
        )
        return x_kp1, (x_k, y_k)

    _, (xs, ys) = jax.lax.scan(body, x1, jax.random.split(sampling_key, T))
    return xs, ys, inv_chol_P0, inv_chol_Q

    
@jax.jit
def log_potential(x, y):

    D = y.shape[-1]

    @partial(jnp.vectorize, signature="(k),(n)->()")
    def _log_potential(x, y):
        Sigma = make_covariance_matrix(x, D)

        chol_Sigma = jnp.linalg.cholesky(Sigma)
        _chol_Sigma_inv = solve_triangular(chol_Sigma, jnp.eye(Sigma.shape[0]), lower=True)
        val = mvn_logpdf(y, jnp.zeros(Sigma.shape[0]), None, _chol_Sigma_inv, constant=False)
        # val = norm.logpdf(y, scale=Sigma)

        return jnp.nansum(val)  # in case the scale is infinite, we get nan, but we want 0
    return _log_potential(x, y)


def log_likelihood(x, y):
    return jnp.sum(log_potential(x, y))


def log_pdf(xs, ys, m0, inv_chol_P0, F, b, inv_chol_Q):
    def _logpdf(zs):
        out = mvn_logpdf(zs[0], m0, None, inv_chol_P0, constant=False)
        # pred_xs = F @ zs[:-1] + b
        # pred_xs = jnp.einsum('...j,ij,i->...i', zs[:-1], F, b)
        pred_xs = jnp.einsum('ij,...j->...i', F, zs[:-1]) + b
        out += jnp.sum(mvn_logpdf(zs[1:], pred_xs, None, inv_chol_Q, constant=False))
        out += log_likelihood(zs, ys)
        return out
    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    T = 300
    D = 3

    m0, P0, Q, F, b = get_dynamics(D)

    print(m0.shape)
    print(P0.shape)
    print(Q.shape)
    print(F.shape)
    print(b.shape)

    xs, ys, inv_chol_P0, inv_chol_Q = get_data(key, m0, P0, Q, F, b, D, T)
    print(xs.shape, ys.shape)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(D, 2, figsize=(15, 5*D))
    if D > 1:
        for d in range(D):
            ax[d, 0].plot(xs[:, d])
            ax[d, 0].set_title(f"Eigenvalue {d}")
            ax[d, 1].plot(ys[:, d])
            ax[d, 1].set_title(f"Observations y dimension {d}")
    else:
        ax[0].plot(xs[:, 0])
        ax[0].set_title("Eigenvalue")
        ax[1].plot(ys[:, 0])
        ax[1].set_title(f"Observations y")
    
    plt.tight_layout()
    plt.savefig("experiments/scalable_stochastic_volatility/plots/data.png")


# ====================== OLD ============================

# ... get dynamics() ...
    # h0 = jnp.zeros((D,))
    # d0 = jnp.zeros((D*(D-1) // 2,))
    # m0 = (h0, d0)

    # phi_h = 0.95 * jnp.ones((D,))
    # phi_d = 0.90 * jnp.ones((D*(D-1) // 2,))
    # phi = (phi_h, phi_d)

    # sigma_h = jnp.diag(0.1 * jnp.ones((D,)))
    # sigma_d = jnp.diag(0.05 * jnp.ones((D*(D-1) // 2,)))
    # Q = (sigma_h, sigma_d)

    # h_P0 = (sigma_h**2) / (1 - phi_h**2)
    # d_P0 = (sigma_d**2) / (1 - phi_d**2)
    # P0 = (h_P0, d_P0)

    # return m0, P0, Q, phi


# def D_from_K(K: jnp.ndarray) -> jnp.ndarray:
#     """
#     X is shape K = D*(D-1) + D = D^2
#     Therefore D = sqrt(K)
#     """        
#     K = jnp.asarray(K)
#     D = jnp.sqrt(K)
#     return D.astype(jnp.int32)

# ... get_data()...
    # h0, d0 = m0
    # phi_h, phi_d = phi
    # Sigma_h, Sigma_d = Q

    # if Sigma_h.ndim == 1:
    #     Sigma_h = jnp.diag(Sigma_h)
    # if Sigma_d.ndim == 1:
    #     Sigma_d = jnp.diag(Sigma_d)

    # h_P0, d_P0 = P0

    # chol_h_P0 = jnp.linalg.cholesky(h_P0)
    # chol_d_P0 = jnp.linalg.cholesky(d_P0)
    # chol_Sigma_h = jnp.linalg.cholesky(Sigma_h)
    # chol_Sigma_d = jnp.linalg.cholesky(Sigma_d)

    # inv_chol_h_P0 = solve(chol_h_P0, jnp.eye(h0.shape[0]), assume_a="pos")
    # inv_chol_d_P0 = solve(chol_d_P0, jnp.eye(d0.shape[0]), assume_a="pos")
    # inv_chol_Sigma_h = solve(chol_Sigma_h, jnp.eye(h0.shape[0]), assume_a="pos")
    # inv_chol_Sigma_d = solve(chol_Sigma_d, jnp.eye(d0.shape[0]), assume_a="pos")

    # h1 = jr.multivariate_normal(init_key_1, h0, h_P0)
    # d1 = jr.multivariate_normal(init_key_2, d0, d_P0)

    # x0 = jnp.concatenate([h1, d1])

    # def body(x_k, key_k):
    #     state_key_h, state_key_d, observation_key = jr.split(key_k, 3)
        
    #     # h_k = x_k[:dim]
    #     # d_k = x_k[dim:]

    #     # # make eigenvalues
    #     # lambdas_k = jnp.exp(h_k)

    #     # # make omegas and eigenvectors
    #     # omegas_k = 0.5 * jnp.pi * (1 - jnp.exp(d_k)) / (1 + jnp.exp(d_k))
    #     # G_k = make_givens_matrices(omegas_k, dim)  # (K, D, D)
    #     # P_k = jnp.prod(G_k, axis=0)  # (D, D)
        
    #     # # make covariance matrix and sample y_k
    #     # Sigma_k = P_k @ jnp.diag(lambdas_k) @ P_k.T

    #     # D = D_from_K(x_k.shape[0])
    #     Sigma_k = make_covariance_matrix(x_k, D)
    #     y_k = jr.multivariate_normal(
    #         observation_key,
    #         mean=jnp.zeros((D,)),
    #         cov=Sigma_k
    #     )

    #     # sample next state x_k+1
    #     h_k = x_k[:D]
    #     d_k = x_k[D:]
    #     h_kp1 = h0 + phi_h * (h_k - h0) + jr.multivariate_normal(state_key_h, jnp.zeros(h_k.shape), Sigma_h)
    #     d_kp1 = d0 + phi_d * (d_k - d0) + jr.multivariate_normal(state_key_d, jnp.zeros(d_k.shape), Sigma_d)

    #     x_kp1 = jnp.concatenate([h_kp1, d_kp1])
    #     return x_kp1, (x_k, y_k)

    # _, (xs, ys) = jax.lax.scan(body, x0, jax.random.split(sampling_key, T))
    # return xs, ys, (inv_chol_h_P0, inv_chol_d_P0), (inv_chol_Sigma_h, inv_chol_Sigma_d)


# ... _log_pdf() ... 
        # hs = zs[:, :D]
        # ds = zs[:, D:]

        # h0, d0 = m0
        # phi_h, phi_d = phi
        # inv_chol_P0_h, inv_chol_P0_d = inv_chol_P0
        # inv_chol_Sigma_h, inv_chol_Sigma_d = inv_chol_Sigma

        # out = mvn_logpdf(hs[0], h0, None, inv_chol_P0_h, constant=False)
        # out += mvn_logpdf(ds[0], d0, None, inv_chol_P0_d, constant=False)

        # pred_hs = h0 + phi_h * (hs[:-1] - h0)
        # pred_ds = d0 + phi_d * (ds[:-1] - d0)

        # out += jnp.sum(mvn_logpdf(hs[1:], pred_hs, None, inv_chol_Sigma_h, constant=False))
        # out += jnp.sum(mvn_logpdf(ds[1:], pred_ds, None, inv_chol_Sigma_d, constant=False))
        # out += log_likelihood(zs, ys, D)
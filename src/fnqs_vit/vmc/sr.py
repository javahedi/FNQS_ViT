# src/fnqs_vit/vmc/sr.py

import jax
import jax.numpy as jnp


# -------------------------------------------------------------------
# Compute per-system statistics <O>, <E_L>, <O E_L>, <O O†>
# -------------------------------------------------------------------
def _compute_statistics(O, E_loc):
    """
    O     : (M, P)  complex log-derivatives
    E_loc : (M,)    local energies (real or complex)

    Returns:
        mean_O      : (P,)
        mean_EL     : scalar
        mean_O_EL   : (P,)
        mean_OOstar : (P, P)
    """
    M = O.shape[0]
    w = 1.0 / M

    mean_O     = jnp.sum(O, axis=0) * w                 # <O>
    mean_EL    = jnp.sum(E_loc) * w                     # <E>
    mean_O_EL  = jnp.sum(O * E_loc[:, None], axis=0) * w  # <O E>
    mean_OO    = (O.conj().T @ O) * w                   # <O O†>

    return mean_O, mean_EL, mean_O_EL, mean_OO


# -------------------------------------------------------------------
# Assemble global SR matrices (average over gamma-systems)
# -------------------------------------------------------------------
def compute_sr_matrices(O_all, E_all):
    """
    O_all: list of arrays shape (M_k, P)
    E_all: list of arrays shape (M_k,)

    Returns:
        G : (P,)  SR force vector
        S : (P,P) SR covariance matrix
    """
    R = len(O_all)
    P = O_all[0].shape[1]     # number of parameters (flattened dim)

    G = jnp.zeros((P,), dtype=jnp.float32)
    S = jnp.zeros((P, P), dtype=jnp.float32)

    for O_k, E_k in zip(O_all, E_all):

        mean_O, mean_EL, mean_O_EL, mean_OO = _compute_statistics(O_k, E_k)

        # gradient / force vector
        G_k = 2.0 * jnp.real(mean_O_EL - mean_EL * mean_O)

        # covariance matrix
        S_k = jnp.real(mean_OO - jnp.outer(mean_O.conj(), mean_O))

        G += G_k / R
        S += S_k / R

    return G, S


# -------------------------------------------------------------------
# Solve SR equation δθ = −η (S + λ I)^−1 G
# -------------------------------------------------------------------
def sr_update(G, S, eta=0.01, diag_shift=0.01):
    """
    G : (P,)
    S : (P,P)

    Returns:
        delta_theta : (P,)
    """
    P = G.shape[0]
    S_reg = S + diag_shift * jnp.eye(P)

    delta = -eta * jax.scipy.linalg.solve(S_reg, G, assume_a='pos')
    return delta


# # src/fnqs_vit/vmc/sr.py

# import jax
# import jax.numpy as jnp


# # -------------------------------------------------------------------
# # Compute per-system statistics <O>, <E_L>, <O E_L> over samples
# # -------------------------------------------------------------------
# def _compute_statistics(O, E_loc):
#     """
#     O        : shape (M, P)   log-derivatives for one system
#     E_loc    : shape (M,)     local energies for one system

#     Returns dictionary with:
#       mean_O      : (P,)
#       mean_EL     : scalar
#       mean_O_EL   : (P,)
#       mean_OOstar : (P,P)
#     """
#     M = O.shape[0]
#     w = 1.0 / M

#     mean_O = jnp.sum(O, axis=0) * w
#     mean_EL = jnp.sum(E_loc) * w
#     mean_O_EL = jnp.sum(O * E_loc[:,None], axis=0) * w
#     mean_OO = (O.conj().T @ O) * w  # (P,P)

#     return mean_O, mean_EL, mean_O_EL, mean_OO


# # -------------------------------------------------------------------
# # Multi-system SR assembly: computes full G and S
# # -------------------------------------------------------------------
# def compute_sr_matrices(O_all, E_all):
#     """
#     Parameters
#     ----------
#     O_all : list of length R
#         Each element is O[k] of shape (M_k, P)
#     E_all : list of length R
#         Each element is E[k] of shape (M_k, )

#     Returns
#     -------
#     G : (P,)
#     S : (P,P)
#     """
#     R = len(O_all)
#     P = O_all[0].shape[1]

#     G = jnp.zeros((P,), dtype=jnp.float32)
#     S = jnp.zeros((P, P), dtype=jnp.float32)

#     for O_k, E_k in zip(O_all, E_all):
#         mean_O, mean_EL, mean_O_EL, mean_OO = _compute_statistics(O_k, E_k)

#         # gradient term
#         G_k = 2.0 * jnp.real(mean_O_EL - mean_EL * mean_O)

#         # S_k term
#         S_k = jnp.real(mean_OO - jnp.outer(mean_O.conj(), mean_O))

#         G += G_k / R
#         S += S_k / R

#     return G, S


# # -------------------------------------------------------------------
# # Solve SR linear system: δθ = -η (S + λ I)^{-1} G
# # -------------------------------------------------------------------
# def sr_update(G, S, eta=0.01, diag_shift=0.01):
#     """
#     Performs the SR update.

#     Parameters
#     ----------
#     G : (P,) gradient vector
#     S : (P,P) SR matrix
#     eta : float, learning rate
#     diag_shift : float, λ regularization

#     Returns
#     -------
#     delta_theta : (P,)
#     """
#     P = G.shape[0]
#     S_reg = S + diag_shift * jnp.eye(P)

#     delta = -eta * jax.scipy.linalg.solve(S_reg, G, assume_a='pos')
#     return delta

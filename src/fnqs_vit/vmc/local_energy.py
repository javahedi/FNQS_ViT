# src/fnqs_vit/vmc/local_energy.py
import jax
import jax.numpy as jnp


def compute_local_energy_batch(
    sigma_batch,        # (M, Lx, Ly)
    gamma_scalar,       # float
    logpsi_fn, 
    logpsi_batch,       # (M,)
    edge_array,         # (E,2)
    J1,
    J2,
    params
):
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly
    E = edge_array.shape[0]

    gamma_field = jnp.ones((Lx, Ly)) * gamma_scalar

    sigma_flat = sigma_batch.reshape(M, N)

    # --------------------------------
    # diagonal terms
    # --------------------------------
    i = edge_array[:, 0]
    j = edge_array[:, 1]

    sij = sigma_flat[:, i] * sigma_flat[:, j]
    E_diag = 0.25 * jnp.sum(sij, axis=1)

    # --------------------------------
    # mask: flippable edges
    # --------------------------------
    flippable = (sigma_flat[:, i] != sigma_flat[:, j])  # (M,E)

    # --------------------------------
    # off-diagonal terms
    # --------------------------------
    def flip_edge_all(ii, jj):
        sf = sigma_flat.at[:, ii].mul(-1).at[:, jj].mul(-1)
        s_prop = sf.reshape(M, Lx, Ly)
        lp = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(s_prop)
        return jnp.exp(lp - logpsi_batch)   # (M,)

    # correct lambda syntax
    off_terms = jax.vmap(lambda edge: flip_edge_all(edge[0], edge[1]))(edge_array)
    # shape (E, M)

    off_terms = off_terms.T  # (M,E)

    # --------------------------------
    # Assign J to edges
    # --------------------------------
    num_nn = edge_array.shape[0] // 2
    Jvec = jnp.concatenate([
        jnp.full((num_nn,), J1),
        jnp.full((edge_array.shape[0] - num_nn,), J2)
    ])

    E_off = jnp.sum(0.5 * Jvec * off_terms * flippable, axis=1)

    return E_diag + E_off

# # src/fnqs_vit/vmc/local_energy.py
# # fully vectorized local energy computation for M samples

# import jax
# import jax.numpy as jnp


# def compute_local_energy_batch(
#     sigma_batch,        # (M, Lx, Ly)
#     gamma_scalar,       # float
#     logpsi_fn, 
#     logpsi_batch,       # (M,)
#     nn_edges, 
#     nnn_edges,
#     J1,
#     J2,
#     params
# ):
#     M, Lx, Ly = sigma_batch.shape
#     N = Lx * Ly
#     gamma_field = jnp.ones((Lx, Ly)) * gamma_scalar

#     sigma_flat = sigma_batch.reshape(M, N)

#     # -------------------------
#     # Diagonal contribution
#     # -------------------------
#     nn_i = jnp.array([i for (i, j) in nn_edges])
#     nn_j = jnp.array([j for (i, j) in nn_edges])
#     nnn_i = jnp.array([i for (i, j) in nnn_edges])
#     nnn_j = jnp.array([j for (i, j) in nnn_edges])

#     # NN diagonal: sum 0.25 * s_i s_j
#     diag_nn = 0.25 * sigma_flat[:, nn_i] * sigma_flat[:, nn_j]
#     diag_nnn = 0.25 * sigma_flat[:, nnn_i] * sigma_flat[:, nnn_j]

#     E_diag = jnp.sum(diag_nn, axis=1) + jnp.sum(diag_nnn, axis=1)

#     # -------------------------
#     # Off-diagonal contribution
#     # -------------------------
#     # Determine which NN edges are flippable
#     mask_nn = (sigma_flat[:, nn_i] != sigma_flat[:, nn_j])  # (M, nn_edges)

#     def flip_edge(sigma_flat, i, j):
#         return sigma_flat.at[[i, j]].mul(-1)

#     # vmap over edges → vmap over sigma_batch
#     def compute_offdiag_single_edge(i, j):
#         # flip for all M at edge (i,j)
#         sigma_prop = sigma_flat.at[:, [i, j]].mul(jnp.array([-1, -1]))
#         sigma_prop = sigma_prop.reshape(M, Lx, Ly)
#         logpsi_prop = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_prop)
#         return jnp.exp(logpsi_prop - logpsi_batch)

#     # vectorized over edges
#     off_nn = jnp.array([
#         compute_offdiag_single_edge(int(i), int(j))
#         for (i, j) in nn_edges
#     ])  # shape = (num_nn_edges, M)

#     off_nnn = jnp.array([
#         compute_offdiag_single_edge(int(i), int(j))
#         for (i, j) in nnn_edges
#     ])

#     # Only add contributions at flippable edges
#     off_nn = 0.5 * J1 * jnp.sum(off_nn.T * mask_nn, axis=1)
#     off_nnn = 0.5 * J2 * jnp.sum(off_nnn.T * (sigma_flat[:, nnn_i] != sigma_flat[:, nnn_j]), axis=1)

#     return E_diag + off_nn + off_nnn


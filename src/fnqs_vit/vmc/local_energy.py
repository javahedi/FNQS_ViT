# src/fnqs_vit/vmc/local_energy.py

import jax
import jax.numpy as jnp


def compute_local_energy_batch(
    sigma_batch,        # (M, Lx, Ly)
    gamma_scalar,
    logpsi_fn,
    logpsi_batch,       # (M,)
    edge_array,         # (E,2)
    edge_type,          # (E,) 0=NN,1=NNN
    J1,
    J2,
    params
):

    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly

    # scalar → field
    gamma_field = jnp.ones((Lx, Ly)) * gamma_scalar

    # flatten to (M,N)
    sigma_flat = sigma_batch.reshape(M, N)

    # ------------------------------------------------------------
    # DIAGONAL ENERGY
    # ------------------------------------------------------------
    i = edge_array[:, 0]
    j = edge_array[:, 1]

    sij = sigma_flat[:, i] * sigma_flat[:, j]  # (M,E)

    Jvec = jnp.where(edge_type == 0, J1, J2)   # (E,)

    E_diag = jnp.sum(0.25 * Jvec * sij, axis=1)  # (M,)

    # ------------------------------------------------------------
    # FLIPPABLE = opposite spins
    # ------------------------------------------------------------
    flippable = (sigma_flat[:, i] != sigma_flat[:, j])  # (M,E)

    # ------------------------------------------------------------
    # OFF-DIAGONAL: SWAP σ_i <-> σ_j
    # ------------------------------------------------------------

    def swap_one_chain(s, ii, jj):
        # swap on a single chain (s: shape (N,))
        si = s[ii]
        sj = s[jj]
        s2 = s.at[ii].set(sj)
        s2 = s2.at[jj].set(si)
        return s2

    def ratio_for_edge(edge):
        ii, jj = edge

        # apply swap to each chain independently
        sf = jax.vmap(lambda s: swap_one_chain(s, ii, jj))(sigma_flat)  # (M,N)

        # reshape to (M,Lx,Ly)
        s_prop = sf.reshape(M, Lx, Ly)

        # evaluate logψ
        lp = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(s_prop)

        return jnp.exp(lp - logpsi_batch)   # (M,)

    # off_terms shape = (E,M) → transpose to (M,E)
    off_terms = jax.vmap(ratio_for_edge)(edge_array).T

    # ------------------------------------------------------------
    # OFF energy = (J/2) * R * flippable
    # ------------------------------------------------------------
    E_off = jnp.sum(0.5 * Jvec * off_terms * flippable, axis=1)

    return E_diag + E_off



# # src/fnqs_vit/vmc/local_energy.py
# import jax
# import jax.numpy as jnp


# def compute_local_energy_batch(
#     sigma_batch,     # (M, Lx, Ly)
#     gamma_scalar,    # float
#     logpsi_fn,
#     logpsi_batch,    # (M,)
#     edge_array,      # (E,2)
#     edge_type,       # (E,)  0 = NN, 1 = NNN
#     J1,
#     J2,
#     params
# ):
#     """
#     Computes the local energy for M configurations for a J1–J2 Heisenberg model:

#         H = sum_{<ij>} J1 [ S_i · S_j ]
#           + sum_{<<ij>>} J2 [ S_i · S_j ]

#     Using spin-½ representation:
#         S^z = σ/2,  σ ∈ {+1, -1}
#         S^z S^z = (σ_i σ_j)/4
#         S^+ S^- + S^- S^+ causes |↑↓> ↔ |↓↑> exchange
#     """

#     M, Lx, Ly = sigma_batch.shape
#     N = Lx * Ly

#     # Uniform gamma field
#     gamma_field = jnp.ones((Lx, Ly)) * gamma_scalar

#     # Flatten spins
#     sigma_flat = sigma_batch.reshape(M, N)

#     # ------------------------------------------------------------
#     # Extract edge list and assign J values correctly
#     # ------------------------------------------------------------
#     i = edge_array[:, 0]   # (E,)
#     j = edge_array[:, 1]

#     # J1 for NN edges, J2 for NNN edges
#     Jvec = jnp.where(edge_type == 0, J1, J2)   # (E,)

#     # ------------------------------------------------------------
#     # DIAGONAL ENERGY:       J * (σ_i σ_j)/4
#     # ------------------------------------------------------------
#     sij = sigma_flat[:, i] * sigma_flat[:, j]     # (M,E)
#     E_diag = jnp.sum(0.25 * Jvec * sij, axis=1)   # (M,)

#     # ------------------------------------------------------------
#     # Flippable edges = opposite spins
#     # ------------------------------------------------------------
#     flippable = (sigma_flat[:, i] != sigma_flat[:, j])  # (M,E)

#     # ------------------------------------------------------------
#     # OFF-DIAGONAL OPERATOR: SWAP spins on edge (i,j)
#     #
#     # |↑↓> ↔ |↓↑>
#     #
#     # H_off contributes:
#     #       (J/2) * R
#     #
#     # where R = ψ(s') / ψ(s)
#     # ------------------------------------------------------------
#     # def swap_edge_all(ii, jj):
#     #     si = sigma_flat[:, ii]  # (M,)
#     #     sj = sigma_flat[:, jj]  # (M,)

#     #     # swap for each chain individually
#     #     sf = sigma_flat.at[:, ii].set(sj)
#     #     sf = sf.at[:, jj].set(si)

#     #     s_prop = sf.reshape(M, Lx, Ly)
#     #     lp = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(s_prop)

#     #     return jnp.exp(lp - logpsi_batch)

#     # # Vectorize over edges
#     # off_terms = jax.vmap(lambda edge: swap_edge_all(edge[0], edge[1]))(edge_array)
#     # off_terms = off_terms.T   # (M,E)

#     # ------------------------------------------------------------
#     # OFF-DIAGONAL ENERGY: swap σ_i and σ_j
#     # ------------------------------------------------------------

#     def swap_one_chain(s, ii, jj):
#         """Swap spins at ii and jj for a single chain."""
#         si = s[ii]
#         sj = s[jj]
#         s2 = s.at[ii].set(sj)
#         s2 = s2.at[jj].set(si)
#         return s2


#     def compute_ratio_for_edge(edge):
#         ii, jj = edge

#         # perform swap for all M chains
#         sf = jax.vmap(lambda s: swap_one_chain(s, ii, jj))(sigma_flat)

#         # reshape to (M, Lx, Ly)
#         s_prop = sf.reshape(M, Lx, Ly)

#         lp = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(s_prop)

#         return jnp.exp(lp - logpsi_batch)   # (M,)


#     # off_terms shape = (E, M)
#     off_terms = jax.vmap(compute_ratio_for_edge)(edge_array).T   # (M, E)


#     # ------------------------------------------------------------
#     # OFF-DIAGONAL ENERGY
#     #       (J/2) * R * flippable
#     # ------------------------------------------------------------
#     E_off = jnp.sum(0.5 * Jvec * off_terms * flippable, axis=1)

#     # ------------------------------------------------------------
#     # TOTAL LOCAL ENERGY
#     # ------------------------------------------------------------
#     return E_diag + E_off

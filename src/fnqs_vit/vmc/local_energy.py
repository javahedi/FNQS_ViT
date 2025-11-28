import jax
import jax.numpy as jnp

def compute_local_energy(
    sigma,
    gamma,
    logpsi_fn,
    logpsi_sigma,
    nn_edges,
    nnn_edges,
    J1,
    J2,
):
    Lx, Ly = sigma.shape
    N = Lx * Ly

    sigma_flat = sigma.reshape(N)

    E_diag = 0.0
    E_offdiag = 0.0 + 0j

    # --- NN edges ---
    for (i,j) in nn_edges:
        Jij = J1

        # diagonal Sz Sz
        E_diag += 0.25 * sigma_flat[i] * sigma_flat[j]

        # off-diagonal flips
        if sigma_flat[i] != sigma_flat[j]:
            sigma_prop = sigma_flat.at[[i,j]].mul(-1)
            sigma_prop = sigma_prop.reshape((Lx, Ly))

            logpsi_prop = logpsi_fn(sigma_prop, gamma)
            ratio = jnp.exp(logpsi_prop - logpsi_sigma)
            E_offdiag += 0.5 * Jij * ratio

    # --- NNN edges ---
    for (i,j) in nnn_edges:
        Jij = J2

        E_diag += 0.25 * sigma_flat[i] * sigma_flat[j]

        if sigma_flat[i] != sigma_flat[j]:
            sigma_prop = sigma_flat.at[[i,j]].mul(-1)
            sigma_prop = sigma_prop.reshape((Lx, Ly))

            logpsi_prop = logpsi_fn(sigma_prop, gamma)
            ratio = jnp.exp(logpsi_prop - logpsi_sigma)
            E_offdiag += 0.5 * Jij * ratio

    return E_diag + E_offdiag

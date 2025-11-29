
# src/fnqs_vit/vmc/sampler.py
# fully vectorized sampler for M independent chains
import jax
import jax.numpy as jnp

# ------------------------------------------------------------
# Create M independent spin chains
# ------------------------------------------------------------
def random_spin_state_batch(key, M, Lx, Ly):
    return jax.random.choice(
        key, 
        jnp.array([-1, 1]), 
        shape=(M, Lx, Ly)
    )


# ------------------------------------------------------------
# One Metropolis update for all M chains in parallel
# ------------------------------------------------------------
def metropolis_update_batch(key, sigma_batch, gamma_field, logpsi_batch, logpsi_fn, params):
    """
    sigma_batch:  (M, Lx, Ly)
    logpsi_batch: (M,)
    gamma_field:  (Lx, Ly)
    """
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly

    # Random sites for each chain
    key, key_idx, key_rand = jax.random.split(key, 3)
    idx = jax.random.randint(key_idx, (M,), 0, N)

    i = idx // Ly
    j = idx % Ly

    # Proposed sigma
    # Build indexing mask
    mask = jax.vmap(lambda ii, jj: jnp.zeros((Lx, Ly), dtype=bool).at[ii, jj].set(True))(i, j)
    sigma_prop_batch = jnp.where(mask, -sigma_batch, sigma_batch)

    # compute logψ for proposed sigmas
    logpsi_prop = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_prop_batch)

    # acceptance
    logA = 2 * (jnp.real(logpsi_prop) - jnp.real(logpsi_batch))
    accept = jax.random.uniform(key_rand, (M,)) < jnp.exp(logA)

    # accept or reject each chain
    accept = accept[:, None, None]   # broadcast
    sigma_new = jnp.where(accept, sigma_prop_batch, sigma_batch)

    accept_lpsi = accept.squeeze()
    logpsi_new = jnp.where(accept_lpsi, logpsi_prop, logpsi_batch)

    return sigma_new, logpsi_new, key


# ------------------------------------------------------------
# One sweep = N sequential updates, all vectorized over M
# ------------------------------------------------------------
def metropolis_sweep_batch(key, sigma_batch, gamma_field, logpsi_batch, logpsi_fn, params):
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly

    def body(carry, _):
        sigma, logpsi, key = carry
        return metropolis_update_batch(key, sigma, gamma_field, logpsi, logpsi_fn, params), None

    (sigma_out, logpsi_out, key_out), _ = jax.lax.scan(
        body,
        (sigma_batch, logpsi_batch, key),
        None,
        length=N
    )
    return sigma_out, logpsi_out, key_out


# ------------------------------------------------------------
# Full sampling: M parallel chains
# ------------------------------------------------------------
def sample_chain_batch(
    key, 
    logpsi_fn, params, 
    gamma_field, 
    sigma_init_batch, 
    n_discard, n_samples
):
    """
    sigma_init_batch: (M, Lx, Ly)
    returns:
        samples:     (n_samples, M, Lx, Ly)
        logpsi_vals: (n_samples, M)
    """

    logpsi_batch = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_init_batch)

    # burn-in
    for _ in range(n_discard):
        sigma_init_batch, logpsi_batch, key = metropolis_sweep_batch(
            key, sigma_init_batch, gamma_field, logpsi_batch, logpsi_fn, params
        )

    # sampling
    def sample_step(carry, _):
        sigma, logpsi, key = carry
        sigma, logpsi, key = metropolis_sweep_batch(key, sigma, gamma_field, logpsi, logpsi_fn, params)
        return (sigma, logpsi, key), (sigma, logpsi)

    (sigma_final, logpsi_final, key_out), (sigma_hist, logpsi_hist) = jax.lax.scan(
        sample_step,
        (sigma_init_batch, logpsi_batch, key),
        None,
        length=n_samples
    )

    return sigma_hist, logpsi_hist


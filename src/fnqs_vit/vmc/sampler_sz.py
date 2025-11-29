# src/fnqs_vit/vmc/sampler_sz.py

import jax
import jax.numpy as jnp


# -------------------------------------------------------------------
# Utility: project a random config into exact S^z sector
# -------------------------------------------------------------------
def initialize_sector(key, M, Lx, Ly, Sztarget):
    """
    Produce M configurations with exact 
       sum_i σ_i = 2 * Sztarget
    using a fast balanced assignment.
    """
    N = Lx * Ly
    target_sum = 2 * Sztarget

    # number of +1 spins:
    n_up = (N + target_sum) // 2
    n_down = N - n_up

    def make_one(k):
        # shuffle
        perm = jax.random.permutation(k, N)
        # first n_up -> +1, next n_down -> -1
        spins = jnp.where(
            jnp.arange(N)[perm] < n_up,
            1,
            -1
        )
        return spins.reshape((Lx, Ly))

    keys = jax.random.split(key, M)
    return jax.vmap(make_one)(keys)


# -------------------------------------------------------------------
# Global pair-flip proposal (σ_i = +1, σ_j = -1)
# -------------------------------------------------------------------
def propose_pair_flips(key, sigma_batch):
    """
    sigma_batch: (M, Lx, Ly)
    returns:
        idx_plus  (M,) flattened positions of +1 sites
        idx_minus (M,) flattened positions of -1 sites
    """
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly
    flat = sigma_batch.reshape(M, N)

    # indices of all +1 and -1 spins
    plus_mask  = (flat ==  1)
    minus_mask = (flat == -1)

    def sample_site(k, mask):
        # number of valid choices per chain varies;
        # use masked uniform sampling:
        probs = mask / mask.sum()
        idx = jax.random.choice(k, N, p=probs)
        return idx

    key_plus, key_minus = jax.random.split(key)
    keys_p = jax.random.split(key_plus,  M)
    keys_m = jax.random.split(key_minus, M)

    idx_plus  = jax.vmap(sample_site)(keys_p, plus_mask)
    idx_minus = jax.vmap(sample_site)(keys_m, minus_mask)

    return idx_plus, idx_minus


# -------------------------------------------------------------------
# Metropolis update with pair flips (sector-conserving)
# -------------------------------------------------------------------
def metropolis_update_pair(key, sigma_batch, gamma_field,
                           logpsi_batch, logpsi_fn, params):
    """
    sigma_batch: (M, Lx, Ly)
    gamma_field: (Lx, Ly)
    logpsi_batch: (M,)
    """
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly
    key, key_prop = jax.random.split(key)

    # choose a pair (i,j)
    idx_plus, idx_minus = propose_pair_flips(key_prop, sigma_batch)

    # Build masks for flipping those positions
    def build_mask(pos):
        m = jnp.zeros((Lx, Ly), bool)
        i = pos // Ly
        j = pos % Ly
        return m.at[i, j].set(True)

    mask_plus  = jax.vmap(build_mask)(idx_plus)    # (M,Lx,Ly)
    mask_minus = jax.vmap(build_mask)(idx_minus)   # (M,Lx,Ly)

    # full flip mask
    mask = jnp.logical_or(mask_plus, mask_minus)   # (M,Lx,Ly)

    sigma_prop = jnp.where(mask, -sigma_batch, sigma_batch)

    # evaluate wavefunction
    logpsi_prop = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_prop)

    # acceptance
    dlog = jnp.real(logpsi_prop - logpsi_batch) * 2.0
    accept = jax.random.uniform(key, (M,)) < jnp.exp(dlog)
    accept = accept[:, None, None]

    sigma_new = jnp.where(accept, sigma_prop, sigma_batch)
    logpsi_new = jnp.where(accept.squeeze(), logpsi_prop, logpsi_batch)

    return sigma_new, logpsi_new, key


# -------------------------------------------------------------------
# Full sweep with sector-preserving pair updates
# -------------------------------------------------------------------
def metropolis_sweep_pair(key, sigma_batch, gamma_field,
                          logpsi_batch, logpsi_fn, params):
    M, Lx, Ly = sigma_batch.shape
    N = Lx * Ly

    def body(carry, _):
        σ, logψ, key = carry
        σ_new, logψ_new, key = metropolis_update_pair(
            key, σ, gamma_field, logψ, logpsi_fn, params
        )
        return (σ_new, logψ_new, key), None

    (σ_fin, logψ_fin, key_fin), _ = jax.lax.scan(
        body,
        (sigma_batch, logpsi_batch, key),
        None,
        length=N,
    )
    return σ_fin, logψ_fin, key_fin


# -------------------------------------------------------------------
# Public API: sample chains in fixed S^z sector
# -------------------------------------------------------------------
def sample_chain_batch_sz(
    key,
    logpsi_fn,
    params,
    gamma_field,
    sigma_init_batch,
    n_discard,
    n_samples
):
    """
    sigma_init_batch: (M,Lx,Ly)
    """
    # compute ψ for initial batch
    logpsi = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_init_batch)

    # burn-in
    for _ in range(n_discard):
        sigma_init_batch, logpsi, key = metropolis_sweep_pair(
            key, sigma_init_batch, gamma_field, logpsi, logpsi_fn, params
        )

    # production
    def step(carry, _):
        σ, logψ, key = carry
        σ, logψ, key = metropolis_sweep_pair(
            key, σ, gamma_field, logψ, logpsi_fn, params
        )
        return (σ, logψ, key), (σ, logψ)

    (_, _, key_out), (σ_hist, logψ_hist) = jax.lax.scan(
        step,
        (sigma_init_batch, logpsi, key),
        None,
        length=n_samples
    )
    return σ_hist, logψ_hist

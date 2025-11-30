# src/fnqs_vit/vmc/sampler_heisenberg.py
import jax
import jax.numpy as jnp


# ------------------------------------------------------------
# Build list of edges (NN + NNN) as a single array
# ------------------------------------------------------------
# def prepare_edge_array(nn_edges, nnn_edges):
#     """
#     Convert Python lists of edges into a single JAX array of shape (E,2),
#     where each row is (i,j) in flattened indexing.
#     """
#     all_edges = nn_edges + nnn_edges
#     return jnp.array(all_edges, dtype=jnp.int32)    # (E,2)
    
def prepare_edge_array(nn_edges, nnn_edges):
    """
    Returns:
        edges:     (E,2) int32
        edge_type: (E,) int8   == 0 for NN, 1 for NNN
    """
    edges = nn_edges + nnn_edges
    edge_type = [0] * len(nn_edges) + [1] * len(nnn_edges)
    return jnp.array(edges, dtype=jnp.int32), jnp.array(edge_type, dtype=jnp.int8)


# ------------------------------------------------------------
# Random initialization
# ------------------------------------------------------------
def random_spin_state_batch(key, M, Lx, Ly):
    return jax.random.choice(key, jnp.array([-1,1]), shape=(M,Lx,Ly))


def random_spin_state_in_sector(key, M, Lx, Ly, Sztarget):
    """
    Initialize *exact Sz sector*: sum σ = 2*Sztarget.
    """
    N = Lx*Ly
    target_sum = 2*Sztarget
    n_up = (N + target_sum)//2   # number of +1 spins

    def make_one(k):
        perm = jax.random.permutation(k, N)
        spins = jnp.where(jnp.arange(N)[perm] < n_up, 1, -1)
        return spins.reshape(Lx, Ly)

    keys = jax.random.split(key, M)
    return jax.vmap(make_one)(keys)


# ------------------------------------------------------------
# One Metropolis update: 2–spin flip on (i,j)
# ------------------------------------------------------------
def metropolis_update_edges(
    key,
    sigma_batch,       # (M,Lx,Ly)
    logpsi_batch,      # (M,)
    gamma_field,       # (Lx,Ly)
    logpsi_fn,
    params,
    edge_array,        # (E,2)
    Lx, Ly,
    restrict_flippable=False,
    p_single=0.90,     # probability for single-spin flip
    p_pair=0.09,       # probability for pair flip
    p_global=0.01      # probability for global Z2 flip
):
    """
    Perform one update on all M chains using a mixture of:
      0 = single-spin flip
      1 = pair flip on (i,j)
      2 = global Z2 flip (sigma -> -sigma)
    """
    M = sigma_batch.shape[0]
    E = edge_array.shape[0]

    # -----------------------------
    # Draw update type per chain
    # -----------------------------
    key, key_mode, key_single, key_edge, key_u = jax.random.split(key, 5)

    probs = jnp.array([p_single, p_pair, p_global])
    cum = jnp.cumsum(probs)
    r = jax.random.uniform(key_mode, (M,))
    mode = jnp.sum(r[:, None] > cum[None, :], axis=1)   # shape (M,)

    # ============================================================
    # (1) SINGLE-SPIN FLIP
    # ============================================================
    N = Lx * Ly
    idx_single = jax.random.randint(key_single, (M,), 0, N)
    ii = idx_single // Ly
    jj = idx_single % Ly

    mask_single = jnp.zeros((M, Lx, Ly), bool)
    mask_single = mask_single.at[jnp.arange(M), ii, jj].set(True)
    sigma_single = jnp.where(mask_single, -sigma_batch, sigma_batch)

    # ============================================================
    # (2) PAIR FLIP (your original update)
    # ============================================================
    edge_idx = jax.random.randint(key_edge, (M,), 0, E)
    ij_pairs = edge_array[edge_idx]              # (M,2)
    i = ij_pairs[:,0]
    j = ij_pairs[:,1]

    def build_mask(pos):
        ii = pos // Ly
        jj = pos % Ly
        m = jnp.zeros((Lx,Ly), bool)
        return m.at[ii, jj].set(True)

    mask_i = jax.vmap(build_mask)(i)
    mask_j = jax.vmap(build_mask)(j)
    flip_mask_pair = jnp.logical_or(mask_i, mask_j)
    sigma_pair = jnp.where(flip_mask_pair, -sigma_batch, sigma_batch)

    # flippability only applies to pair-flip mode
    if restrict_flippable:
        flat = sigma_batch.reshape(M, N)
        si = flat[jnp.arange(M), i]
        sj = flat[jnp.arange(M), j]
        flippable_pair = (si != sj)
        flippable = jnp.where(mode == 1, flippable_pair, True)
    else:
        flippable = jnp.ones((M,), bool)

    # ============================================================
    # (3) GLOBAL Z2 FLIP
    # ============================================================
    sigma_global = -sigma_batch

    # ============================================================
    # SELECT PROPOSAL
    # ============================================================
    # stack all as (3, M, Lx, Ly)
    sigma_all = jnp.stack([sigma_single, sigma_pair, sigma_global], axis=0)
    sigma_prop = sigma_all[mode, jnp.arange(M)]

    # ------------------------------------------------------------
    # Compute logpsi for proposal
    # ------------------------------------------------------------
    logpsi_prop = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_prop)

    # ------------------------------------------------------------
    # Metropolis acceptance
    # ------------------------------------------------------------
    dlog = 2.0 * (jnp.real(logpsi_prop) - jnp.real(logpsi_batch))
    u = jax.random.uniform(key_u, (M,))
    accept = (u < jnp.exp(dlog)) & flippable
    accept_mask = accept[:, None, None]

    sigma_new = jnp.where(accept_mask, sigma_prop, sigma_batch)
    logpsi_new = jnp.where(accept, logpsi_prop, logpsi_batch)

    return sigma_new, logpsi_new, key


# ------------------------------------------------------------
# One sweep over all edges
# ------------------------------------------------------------
def metropolis_sweep_edges(
    key,
    sigma_batch,
    logpsi_batch,
    gamma_field,
    logpsi_fn,
    params,
    edge_array,
    Lx, Ly,
    restrict_flippable=False
):
    Nupdates = edge_array.shape[0]  # E edges

    def body(carry, _):
        sigma, logpsi, key = carry
        sigma, logpsi, key = metropolis_update_edges(
            key, sigma, logpsi, gamma_field, logpsi_fn, params,
            edge_array, Lx, Ly,
            restrict_flippable=restrict_flippable
        )
        return (sigma, logpsi, key), None

    (sigma_out, logpsi_out, key_out), _ = jax.lax.scan(
        body,
        (sigma_batch, logpsi_batch, key),
        None,
        length=Nupdates
    )
    return sigma_out, logpsi_out, key_out


# ------------------------------------------------------------
# Public sampling API (handles Sz or full space)
# ------------------------------------------------------------
def sample_chain_batch_edges(
    key,
    logpsi_fn,
    params,
    gamma_field,
    sigma_init_batch,
    n_discard,
    n_samples,
    edge_array,
    Lx, Ly,
    Sztarget=None
):
    """
    Returns:
        sigma_hist: (n_samples, M, Lx, Ly)
        logpsi_hist: (n_samples, M)
    """

    restrict = (Sztarget is not None)

    # initial logpsi
    logpsi = jax.vmap(lambda s: logpsi_fn(params, s, gamma_field))(sigma_init_batch)

    # burn-in
    for _ in range(n_discard):
        sigma_init_batch, logpsi, key = metropolis_sweep_edges(
            key, sigma_init_batch, logpsi, gamma_field,
            logpsi_fn, params,
            edge_array, Lx, Ly,
            restrict_flippable=restrict
        )

    # production
    def step(carry, _):
        σ, logψ, key = carry
        σ, logψ, key = metropolis_sweep_edges(
            key, σ, logψ, gamma_field,
            logpsi_fn, params,
            edge_array, Lx, Ly,
            restrict_flippable=restrict
        )
        return (σ, logψ, key), (σ, logψ)

    (_, _, _), (σ_hist, logψ_hist) = jax.lax.scan(
        step,
        (sigma_init_batch, logpsi, key),
        None,
        length=n_samples
    )

    return σ_hist, logψ_hist

# src/fnqs_vit/vmc/sampler.py

import jax
import jax.numpy as jnp


# -------------------------------------------------------------
# Utility: Random initial spin configuration
# -------------------------------------------------------------
def random_spin_state(key, Lx, Ly):
    """
    Generates a random σ ∈ {-1, +1}^{Lx×Ly}.
    """
    spins = jax.random.choice(
        key, jnp.array([-1, 1]), shape=(Lx, Ly)
    )
    return spins


# -------------------------------------------------------------
# Utility: Compute log|ψ|² difference for a flip
# -------------------------------------------------------------
def log_prob_diff(logpsi_old, logpsi_new):
    """
    Δlog prob = log |ψ_new|² - log |ψ_old|²
               = 2 * (Re[logψ_new] - Re[logψ_old])
    """
    return 2.0 * (jnp.real(logpsi_new) - jnp.real(logpsi_old))


# -------------------------------------------------------------
# Metropolis-Hastings single update
# -------------------------------------------------------------
def metropolis_step(key, sigma, gamma, logpsi_fn, logpsi_sigma):
    """
    Perform a single Metropolis update by flipping one random spin.

    Parameters
    ----------
    key : jax.random.PRNGKey
    sigma : jnp.ndarray (Lx, Ly)
    gamma : γ parameters (scalar or array)
    logpsi_fn : function(sigma, gamma) -> complex scalar
    logpsi_sigma : logψ(sigma, gamma)

    Returns
    -------
    sigma_new, logpsi_new
    """

    Lx, Ly = sigma.shape
    N = Lx * Ly

    # pick random site to flip
    key, k1, k2 = jax.random.split(key, 3)

    idx = jax.random.randint(k1, (), 0, N)
    ix = idx // Ly
    iy = idx % Ly

    # propose flip
    sigma_prop = sigma.at[ix, iy].mul(-1)

    # compute new logψ
    logpsi_prop = logpsi_fn(sigma_prop, gamma)

    # acceptance
    dlog = log_prob_diff(logpsi_sigma, logpsi_prop)
    accept = jnp.log(jax.random.uniform(k2)) < dlog

    sigma_new = jnp.where(accept, sigma_prop, sigma)
    logpsi_new = jnp.where(accept, logpsi_prop, logpsi_sigma)

    return sigma_new, logpsi_new, key


# -------------------------------------------------------------
# Sweep (N attempts = N sites)
# -------------------------------------------------------------
def metropolis_sweep(key, sigma, gamma, logpsi_fn, logpsi_sigma):
    """
    One full sweep of N Metropolis steps.

    Returns updated (sigma, logpsi_sigma, key).
    """

    def step(carry, _):
        sigma_c, logpsi_c, key_c = carry
        sigma_c, logpsi_c, key_c = metropolis_step(key_c, sigma_c, gamma, logpsi_fn, logpsi_c)
        return (sigma_c, logpsi_c, key_c), None

    N = sigma.size  # number of sites
    (sigma_new, logpsi_new, key_new), _ = jax.lax.scan(
        step,
        (sigma, logpsi_sigma, key),
        xs=None,
        length=N,
    )

    return sigma_new, logpsi_new, key_new


# -------------------------------------------------------------
# Collect M samples after n_discard sweeps
# -------------------------------------------------------------
def sample_chain(key, logpsi_fn, gamma, sigma_init,
                 n_discard, n_samples):
    """
    Draw n_samples configurations after discarding n_discard sweeps.

    Returns:
        (samples, logpsi_values)
    """
    sigma = sigma_init
    logpsi_sigma = logpsi_fn(sigma, gamma)

    # burn-in
    for _ in range(n_discard):
        sigma, logpsi_sigma, key = metropolis_sweep(key, sigma, gamma, logpsi_fn, logpsi_sigma)

    # collect samples
    samples = []
    logpsi_vals = []

    for _ in range(n_samples):
        sigma, logpsi_sigma, key = metropolis_sweep(key, sigma, gamma, logpsi_sigma)

        samples.append(sigma)
        logpsi_vals.append(logpsi_sigma)

    return jnp.stack(samples), jnp.stack(logpsi_vals)


# -------------------------------------------------------------
# Batch sampler over multiple γ values
# -------------------------------------------------------------
def sample_many_gammas(key, logpsi_fn, gammas, Lx, Ly,
                       n_discard=10, n_samples=100):
    """
    Perform independent sampling for each γ in gammas.

    gammas: array of shape (R, ...)
    Returns dict with:
      "samples": (R, n_samples, Lx, Ly)
      "logpsi": (R, n_samples)
    """

    R = len(gammas)
    keys = jax.random.split(key, R)

    def run_single(args):
        key_i, gamma_i = args
        sigma0 = random_spin_state(key_i, Lx, Ly)
        samples, logpsi_vals = sample_chain(
            key_i, logpsi_fn, gamma_i, sigma0,
            n_discard, n_samples
        )
        return samples, logpsi_vals

    samples_all, logpsi_all = jax.vmap(run_single)((keys, gammas))

    return {
        "samples": samples_all,
        "logpsi": logpsi_all,
    }

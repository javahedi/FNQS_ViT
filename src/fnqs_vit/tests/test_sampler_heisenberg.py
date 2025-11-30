import jax
import jax.numpy as jnp

from fnqs_vit.vmc.sampler_heisenberg import (
    random_spin_state_batch,
    metropolis_sweep_edges,
    prepare_edge_array,
)


# ================================================================
# Fake wavefunctions for testing
# ================================================================
def logpsi_constant(params, sigma, gamma):
    """ψ(s) = 1 → logψ = 0"""
    return 0.0


def logpsi_ferromagnetic(params, sigma, gamma):
    """ψ(s) = exp(α * sum σ)"""
    alpha = params["alpha"]
    return alpha * jnp.sum(sigma)


# ================================================================
# Helper to build a trivial 1D chain of size L
# ================================================================
def simple_edges(L):
    """Linear chain 0-1-2-...(L-1)."""
    edges = [(i, i + 1) for i in range(L - 1)]
    return prepare_edge_array(edges, [])  # returns (edges, edge_type)


# ================================================================
# Test 1: constant wavefunction → sampler should move freely
# ================================================================
def test_sampler_constant_wavefunction():
    key = jax.random.PRNGKey(0)
    M = 5
    Lx, Ly = 4, 4
    gamma = jnp.ones((Lx, Ly))

    sigma = random_spin_state_batch(key, M, Lx, Ly)
    logpsi = jnp.zeros((M,))  # logψ = 0

    edges, _ = simple_edges(Lx * Ly)  # ignore edge_type

    # run 20 sweeps
    s0 = sigma.copy()
    for _ in range(20):
        sigma, logpsi, key = metropolis_sweep_edges(
            key, sigma, logpsi,
            gamma,
            logpsi_constant, {},
            edges,
            Lx, Ly,
            restrict_flippable=False,
        )

    # Check: spins must have changed
    assert jnp.any(sigma != s0), "Sampler did not move for constant wavefunction"


# ================================================================
# Test 2: ferromagnetic wavefunction → sampler biases to ↑↑…↑
# ================================================================
def test_sampler_ferromagnetic():
    key = jax.random.PRNGKey(1)
    M = 4
    Lx, Ly = 3, 3
    gamma = jnp.ones((Lx, Ly))

    params = {"alpha": 2.0}  # strong ferromagnetic bias

    sigma = random_spin_state_batch(key, M, Lx, Ly)
    logpsi = jax.vmap(lambda s: logpsi_ferromagnetic(params, s, gamma))(sigma)

    edges, _ = simple_edges(Lx * Ly)

    # perform several sweeps
    for _ in range(30):
        sigma, logpsi, key = metropolis_sweep_edges(
            key, sigma, logpsi,
            gamma,
            logpsi_ferromagnetic, params,
            edges,
            Lx, Ly,
            restrict_flippable=False,
        )

    # All chains should be mostly +1
    final_avg = jnp.mean(sigma)
    assert final_avg > 0.8, f"Ferromagnetic sampler failed: avg = {final_avg}"


# ================================================================
# Test 3: detailed balance in 1×2 system (enumerable)
# ================================================================
def test_detailed_balance_two_site():
    key = jax.random.PRNGKey(2)
    M = 50
    Lx, Ly = 1, 2
    gamma = jnp.ones((Lx, Ly))

    edges, _ = prepare_edge_array([(0, 1)], [])

    params = {"alpha": 1.0}

    sigma = random_spin_state_batch(key, M, Lx, Ly)
    logpsi = jax.vmap(lambda s: logpsi_ferromagnetic(params, s, gamma))(sigma)

    # sweep enough times to thermalize
    for _ in range(50):
        sigma, logpsi, key = metropolis_sweep_edges(
            key, sigma, logpsi,
            gamma,
            logpsi_ferromagnetic, params,
            edges,
            Lx, Ly,
            restrict_flippable=False,
        )

    # empirical marginal distribution:
    # for α>0, p(↑↑) should dominate
    p_upup = jnp.mean((sigma[:, 0, 0] == 1) & (sigma[:, 0, 1] == 1))
    assert p_upup > 0.5, f"DB check failed: p(up,up)={p_upup}"

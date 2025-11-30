import jax
import jax.numpy as jnp

from fnqs_vit.vmc.local_energy import compute_local_energy_batch
from fnqs_vit.vmc.sampler_heisenberg import prepare_edge_array


# ================================================================
# Fake wavefunctions
# ================================================================

def logpsi_const(params, sigma, gamma):
    """ψ(s) = constant"""
    return 0.0


def logpsi_sum(params, sigma, gamma):
    """logψ = α ∑ σ_i"""
    alpha = params["alpha"]
    return alpha * jnp.sum(sigma)


# ================================================================
# Helper: 1D chain edges
# ================================================================

def simple_chain_edges(L):
    edges = [(i, i+1) for i in range(L-1)]
    return prepare_edge_array(edges, [])


# ================================================================
# Test 1 — exact 2-site Heisenberg model
# ================================================================

def test_local_energy_two_site_exact():
    """
    For 2-site Heisenberg:
      H = (J/4) σ1 σ2  +  (J/2)(|↑↓⟩⟨↓↑| + |↓↑⟩⟨↑↓|)

    For ψ = constant:
        ↑↑ ->  +J/4
        ↓↓ ->  +J/4
        ↑↓ ->  (-J/4) + (J/2) = +J/4
        ↓↑ ->  same = +J/4
    """
    J = 1.0
    gamma_scalar = 1.0
    edges, etype = prepare_edge_array([(0, 1)], [])

    sigma_batch = jnp.array([
        [[1,  1]],
        [[-1, -1]],
        [[1, -1]],
        [[-1, 1]],
    ])

    logpsi_batch = jnp.zeros(4)
    params = {}

    E = compute_local_energy_batch(
        sigma_batch,
        gamma_scalar,
        logpsi_const,
        logpsi_batch,
        edges,
        etype,
        J1=J,
        J2=0.0,
        params=params
    )

    assert jnp.allclose(E, 0.25), f"Expected all energies = 0.25, got {E}"


# ================================================================
# Test 2 — constant ψ on a 4-site chain
# ================================================================

def test_local_energy_constant_wavefunction():
    """
    Chain of 4, alternating spins:  ↑↓↑↓

      diagonal:
         σ_i σ_j = [-1, -1, -1]
         E_diag = sum (J/4 * σ_i σ_j)
                 = 3 * (2/4 * -1)
                 = -1.5

      off-diagonal:
         3 flippable bonds, each gives (J/2) = 1
         E_off = 3

      Total = -1.5 + 3 = 1.5
    """
    J = 2.0
    gamma_scalar = 0.7

    edges, etype = simple_chain_edges(4)

    sigma = jnp.array([[1, -1, 1, -1]]).reshape(1, 1, 4)
    logpsi = jnp.zeros((1,))
    params = {}

    E = compute_local_energy_batch(
        sigma,
        gamma_scalar,
        logpsi_const,
        logpsi,
        edges,
        etype,
        J1=J,
        J2=0.0,
        params=params
    )

    expected = -1.5 + 3.0
    assert jnp.allclose(E, expected), f"Expected {expected}, got {E}"


# ================================================================
# Test 3 — ferromagnet vs domain wall (Heisenberg degeneracy)
# ================================================================

def test_local_energy_ferromagnetic_bias():
    """
    For 2-site Heisenberg, ↑↑ and ↑↓ have *equal* energy +1/4
    (triplet degeneracy). Thus E(ferro) == E(domain-wall).
    """
    gamma_scalar = 1.0
    alpha = 3.0
    params = {"alpha": alpha}

    edges, etype = prepare_edge_array([(0, 1)], [])

    sigma_batch = jnp.array([
        [[1,  1]],   # ferro
        [[1, -1]],   # domain wall
    ])

    logpsi_batch = jax.vmap(lambda s: logpsi_sum(params, s, gamma_scalar))(sigma_batch)

    E = compute_local_energy_batch(
        sigma_batch,
        gamma_scalar,
        logpsi_sum,
        logpsi_batch,
        edges,
        etype,
        J1=1.0,
        J2=0.0,
        params=params
    )

    # Correct expectation: degeneracy of triplet sector
    assert jnp.allclose(E[0], E[1]), f"Expected degeneracy, got {E}"


# ================================================================
# Test 4 — batch shape
# ================================================================

def test_local_energy_shape():
    M, Lx, Ly = 10, 3, 3
    sigma_batch = jax.random.choice(
        jax.random.PRNGKey(0),
        jnp.array([-1, 1]),
        shape=(M, Lx, Ly)
    )

    edges, etype = simple_chain_edges(Lx * Ly)

    gamma_scalar = 0.5
    params = {}
    logpsi = jnp.zeros(M)

    E = compute_local_energy_batch(
        sigma_batch,
        gamma_scalar,
        logpsi_const,
        logpsi,
        edges,
        etype,
        J1=1.0,
        J2=0.0,
        params=params
    )

    assert E.shape == (M,)
    assert jnp.isfinite(E).all()

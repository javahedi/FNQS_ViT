import jax.numpy as jnp
import netket as nk
from fnqs_vit.hamiltonians.lattice import create_square_lattice
from fnqs_vit.vmc.local_energy import compute_local_energy


def fake_logpsi(sigma, gamma):
    return jnp.sum(sigma) + 1j*gamma


def test_local_energy_simple():
    L = 2
    gamma = 0.5
    J1 = 1.0
    J2 = 0.5

    nn_edges, nnn_edges = create_square_lattice(L, L)

    sigma = jnp.ones((L,L))
    logpsi = fake_logpsi(sigma, gamma)

    E = compute_local_energy(
        sigma, gamma, fake_logpsi, logpsi,
        nn_edges, nnn_edges, J1, J2
    )

    assert jnp.isfinite(E.real)
    assert jnp.isfinite(E.imag)

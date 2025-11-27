import jax
import jax.numpy as jnp
from fnqs_vit.vmc.sampler import (
    random_spin_state,
    metropolis_step,
    metropolis_sweep,
)


def fake_logpsi(sigma, gamma):
    # simple fake wavefunction: logψ = sum(σ) + i*0
    return jnp.sum(sigma)


def test_metropolis_step():
    key = jax.random.PRNGKey(0)
    sigma = jnp.ones((4,4))
    gamma = 0.5

    logpsi_sigma = fake_logpsi(sigma, gamma)
    sigma_new, logpsi_new, key_new = metropolis_step(
        key, sigma, gamma, fake_logpsi, logpsi_sigma
    )

    assert sigma_new.shape == (4,4)


def test_metropolis_sweep():
    key = jax.random.PRNGKey(1)
    sigma = jnp.ones((2,2))
    gamma = 0.3
    logpsi_sigma = fake_logpsi(sigma, gamma)

    sigma_new, logpsi_new, key_new = metropolis_sweep(
        key, sigma, gamma, fake_logpsi, logpsi_sigma
    )

    assert sigma_new.shape == (2,2)

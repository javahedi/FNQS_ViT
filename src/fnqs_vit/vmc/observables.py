# src/fnqs_vit/vmc/observables.py

import jax
import jax.numpy as jnp
import numpy as np


def _sz_field(samples):
    """
    samples: (M, Lx, Ly) with σ = ±1
    returns Sz: (M, Lx, Ly) with Sz = σ / 2
    """
    return samples / 2.0


def structure_factor_k(samples, kx, ky):
    """
    Compute C(k) = (3/N) < |sum_r e^{ik·r} Sz_r |^2 >

    Parameters
    ----------
    samples : jnp.ndarray, shape (M, Lx, Ly)
        Monte Carlo samples σ = ±1.
    kx, ky : float
        Momentum components in units where lattice spacing = 1.

    Returns
    -------
    Ck : float
        Spin structure factor C(k).
    """
    M, Lx, Ly = samples.shape
    N = Lx * Ly

    sz = _sz_field(samples)  # (M, Lx, Ly)

    # coordinates
    xs = jnp.arange(Lx)
    ys = jnp.arange(Ly)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")  # (Lx, Ly)

    phase = jnp.exp(1j * (kx * X + ky * Y))  # (Lx, Ly)

    # For each sample: Sz_k = sum_r e^{ik·r} Sz_r
    def sz_k_single(sz_cfg):
        return jnp.sum(sz_cfg * phase)

    sz_k = jax.vmap(sz_k_single)(sz)  # (M,)

    # C(k) = (3/N) < |Sz_k|^2 >
    Ck = 3.0 / N * jnp.mean(jnp.abs(sz_k) ** 2)
    return float(Ck)


def m2_neel(samples):
    """
    m_Neel^2 = C(pi, pi) / N
    """
    M, Lx, Ly = samples.shape
    N = Lx * Ly

    C_pi_pi = structure_factor_k(samples, jnp.pi, jnp.pi)
    return C_pi_pi / N


def m2_stripe(samples):
    """
    m_stripe^2 = [C(0,pi) + C(pi,0)] / (2 N)
    """
    M, Lx, Ly = samples.shape
    N = Lx * Ly

    C_0_pi = structure_factor_k(samples, 0.0, jnp.pi)
    C_pi_0 = structure_factor_k(samples, jnp.pi, 0.0)

    return (C_0_pi + C_pi_0) / (2.0 * N)

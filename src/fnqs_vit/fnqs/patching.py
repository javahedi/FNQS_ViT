# src/fnqs_vit/fnqs/patching.py

import jax
import jax.numpy as jnp
from einops import rearrange, asnumpy


def extract_patches_2d(array, patch_size):
    """
    Extract non-overlapping 2D patches from a (Lx, Ly) array.
    
    Parameters
    ----------
    array : jnp.ndarray, shape (Lx, Ly)
        2D lattice configuration (σ or γ).
    patch_size : tuple (px, py)
        Size of each patch.
    
    Returns
    -------
    patches : jnp.ndarray, shape (n_patches, px*py)
        Flattened patches.
    """
    px, py = patch_size
    Lx, Ly = array.shape

    assert Lx % px == 0, "Lx must be divisible by patch size px"
    assert Ly % py == 0, "Ly must be divisible by patch size py"

    # Rearrange into patches using einops
    patches = rearrange(
        array,
        "(nx px) (ny py) -> (nx ny) (px py)",
        px=px, py=py
    )
    return patches


def extract_sigma_patches(sigma, patch_size=(2, 2)):
    """
    Extract patches from physical configuration σ.

    Parameters
    ----------
    sigma : jnp.ndarray, shape (Lx, Ly)
    patch_size : (int, int)

    Returns
    -------
    jnp.ndarray, shape (n_patches, px*py)
    """
    return extract_patches_2d(sigma, patch_size)


def extract_gamma_patches(gamma, patch_size=(2, 2)):
    """
    Extract patches from coupling configuration γ.
    Same shape as σ if γ is site-based.
    
    Parameters
    ----------
    gamma : jnp.ndarray, shape (Lx, Ly) or (Lx, Ly, C)
    patch_size : (int, int)
    
    Returns
    -------
    jnp.ndarray, shape (n_patches, px*py*C)
    """

    # If γ has extra channels (like multiple couplings per site)
    if gamma.ndim == 3:
        px, py = patch_size
        Lx, Ly, C = gamma.shape

        assert Lx % px == 0 and Ly % py == 0

        patches = rearrange(
            gamma,
            "(nx px) (ny py) C -> (nx ny) (px py C)",
            px=px, py=py
        )
        return patches

    # Scalar γ per site
    return extract_patches_2d(gamma, patch_size)

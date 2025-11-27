import jax.numpy as jnp
from fnqs_vit.fnqs.patching import (
    extract_patches_2d,
    extract_sigma_patches,
    extract_gamma_patches,
)

def test_sigma_patches():
    sigma = jnp.arange(16).reshape(4,4)
    patches = extract_sigma_patches(sigma, (2,2))
    
    assert patches.shape == (4, 4)   # 4 patches, each flattened to 4 elements

    # first patch = [[0,1],[4,5]] -> flattened [0,1,4,5]
    assert (patches[0] == jnp.array([0,1,4,5])).all()


def test_gamma_patches_scalar():
    gamma = jnp.ones((4,4))
    patches = extract_gamma_patches(gamma, (2,2))
    assert patches.shape == (4,4)


def test_gamma_patches_channels():
    gamma = jnp.ones((4,4,3))  # 3 couplings per site
    patches = extract_gamma_patches(gamma, (2,2))
    assert patches.shape == (4, 2*2*3)

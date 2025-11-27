import jax
import jax.numpy as jnp
from fnqs_vit.fnqs.embedding import MultimodalEmbedding

def test_small_gamma_embedding():
    sigma = jnp.arange(16).reshape(4,4)
    sigma_patches = jnp.array([
        [0,1,4,5],
        [2,3,6,7],
        [8,9,12,13],
        [10,11,14,15],
    ])

    gamma = jnp.array([0.5])  # O(1)

    model = MultimodalEmbedding(embed_dim=8, gamma_mode="small")

    variables = model.init(jax.random.PRNGKey(0), sigma_patches, gamma)
    out = model.apply(variables, sigma_patches, gamma)

    assert out.shape == (4, 8)

def test_structured_gamma_embedding():
    sigma_patches = jnp.ones((4,4))
    gamma_patches = jnp.ones((4,4*3))  # 3 channels

    model = MultimodalEmbedding(embed_dim=12, gamma_mode="structured")

    variables = model.init(jax.random.PRNGKey(1), sigma_patches, gamma_patches)
    out = model.apply(variables, sigma_patches, gamma_patches)

    assert out.shape == (4, 12)

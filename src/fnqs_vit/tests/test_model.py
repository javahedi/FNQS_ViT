import jax
import jax.numpy as jnp
from fnqs_vit.fnqs.model import FNQSViT


def test_fnqs_vit_forward():
    key = jax.random.PRNGKey(0)

    sigma = jnp.arange(16).reshape(4,4)
    gamma = jnp.ones((4,4))  # structured mode

    model = FNQSViT(
        depth=2,
        embed_dim=12,
        hidden_dim=48,
        num_heads=3,
        patch_size=(2,2),
        gamma_mode="structured",
        translational_invariant=False,
    )

    params = model.init(key, sigma, gamma)
    out = model.apply(params, sigma, gamma)

    assert out.shape == ()          # scalar
    assert jnp.iscomplexobj(out)    # complex

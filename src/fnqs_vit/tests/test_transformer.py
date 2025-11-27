import jax
import jax.numpy as jnp
from fnqs_vit.fnqs.transformer import TransformerEncoder


def test_transformer_encoder():
    key = jax.random.PRNGKey(0)

    n_patches = 4
    dim = 12
    x = jax.random.normal(key, (n_patches, dim))

    model = TransformerEncoder(
        depth=2,
        dim=dim,
        hidden_dim=4 * dim,
        num_heads=3,
        translational_invariant=False
    )

    variables = model.init(key, x)
    out = model.apply(variables, x)

    assert out.shape == (n_patches, dim)

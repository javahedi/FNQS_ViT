import jax.numpy as jnp
from fnqs_vit.vmc.sr import compute_sr_matrices, sr_update


def test_sr_simple():
    # Two systems, tiny P=2
    O1 = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    E1 = jnp.array([1.0, 1.2])

    O2 = jnp.array([[0.5, -0.5], [1.0, 1.0]])
    E2 = jnp.array([0.3, 0.4])

    G, S = compute_sr_matrices([O1, O2], [E1, E2])

    assert G.shape == (2,)
    assert S.shape == (2,2)

    delta = sr_update(G, S, eta=0.01, diag_shift=0.1)
    assert delta.shape == (2,)

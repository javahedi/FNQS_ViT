# src/fnqs_vit/tests/test_observables.py

import jax.numpy as jnp
from fnqs_vit.vmc.observables import m2_neel, m2_stripe

def test_observables_shapes():
    # fake ferromagnetic samples (all +1): Néel and stripe should be ~0
    samples = jnp.ones((10, 4, 4))
    mN = m2_neel(samples)
    mS = m2_stripe(samples)

    assert abs(mN) < 1e-6
    assert abs(mS) < 1e-6

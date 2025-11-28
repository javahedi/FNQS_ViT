# src/fnqs_vit/experiments/compute_observables.py

import pickle
import jax
import jax.numpy as jnp
import numpy as np
import os

from fnqs_vit.fnqs.model import FNQSViT
from fnqs_vit.vmc.sampler import sample_chain, random_spin_state
from fnqs_vit.vmc.observables import m2_neel, m2_stripe


def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_observables_for_gamma(params, model, gamma, Lx, Ly, n_samples=500):

    # define logpsi(theta)(sigma, gamma)
    logpsi_fn = lambda s, g: model.apply(params, s, g)

    # initial state for sampling
    key = jax.random.PRNGKey(123)
    sigma0 = random_spin_state(key, Lx, Ly)

    samples, logpsi_vals = sample_chain(
        key,
        logpsi_fn,
        gamma,
        sigma0,
        n_discard=50,
        n_samples=n_samples,
    )

    # compute observables
    mN2 = m2_neel(samples)
    mS2 = m2_stripe(samples)

    return {
        "m2_neel": float(mN2),
        "m2_stripe": float(mS2),
    }


def main():
    logdir = "logs/fnqs_j1j2"
    params_path = os.path.join(logdir, "params.pkl")

    params = load_params(params_path)

    # same model as training
    L = 6
    model = FNQSViT(
        Lx=L, Ly=L,
        patch_size=(2,2),
        dim=72,
        depth=4,
        hidden_dim=288,
        num_heads=12,
    )

    gammas = jnp.linspace(0.4, 0.6, 10)

    results = {}
    for gamma in gammas:
        print(f"Computing observables for γ={gamma:.3f}")
        obs = compute_observables_for_gamma(params, model, float(gamma), L, L)
        results[float(gamma)] = obs
        print("  ", obs)

    # save
    out = os.path.join(logdir, "observables.json")
    import json
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved observables -> {out}")


if __name__ == "__main__":
    main()

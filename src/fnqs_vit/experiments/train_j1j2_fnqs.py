# src/fnqs_vit/experiments/train_j1j2_fnqs.py

import os
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import pickle



from fnqs_vit.fnqs.model import FNQSViT
from fnqs_vit.hamiltonians.lattice import create_square_lattice
from fnqs_vit.vmc.sampler import sample_many_gammas
from fnqs_vit.vmc.local_energy import compute_local_energy
from fnqs_vit.vmc.sr import compute_sr_matrices, sr_update


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

def get_config():
    return {
        "L": 6,
        "patch_size": (2,2),
        "dim": 72,
        "depth": 4,
        "hidden_dim": 288,
        "num_heads": 12,

        "R": 10,  # number of gamma systems
        "gamma_min": 0.4,
        "gamma_max": 0.6,

        "n_discard": 10,
        "samples_per_system": 200,

        "eta": 0.01,
        "diag_shift": 0.01,
        "iterations": 50,

        "seed": 42,
        "logdir": "logs/fnqs_j1j2",
    }


# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------

def train_fnqs_j1j2():
    cfg = get_config()

    os.makedirs(cfg["logdir"], exist_ok=True)

    # Lattice
    L = cfg["L"]
    Lx, Ly = L, L
    nn_edges, nnn_edges = create_square_lattice(Lx, Ly)

    # Gamma distribution
    gammas = jnp.linspace(cfg["gamma_min"], cfg["gamma_max"], cfg["R"])

    # Model
    key = jax.random.PRNGKey(cfg["seed"])
    model = FNQSViT(
        Lx=Lx,
        Ly=Ly,
        patch_size=cfg["patch_size"],
        dim=cfg["dim"],
        depth=cfg["depth"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
    )

    # Example dummy input for param initialization
    sigma0 = jnp.ones((Lx, Ly))
    gamma0 = 0.5
    params = model.init(key, sigma0, gamma0)

    logpsi_fn = lambda s, g: model.apply(params, s, g)

    # Main training loop
    log = {"energy": []}

    for it in range(cfg["iterations"]):
        t0 = time.time()

        # ---------------------------------------------------------
        # Sampling for all gammas
        # ---------------------------------------------------------
        sample_key = jax.random.split(key, 1)[0]
        samples_info = sample_many_gammas(
            sample_key,
            logpsi_fn,
            gammas,
            Lx,
            Ly,
            cfg["n_discard"],
            cfg["samples_per_system"],
        )

        samples = samples_info["samples"]     # (R, M, Lx, Ly)
        logpsi_vals = samples_info["logpsi"]  # (R, M)

        # ---------------------------------------------------------
        # Compute local energy for each gamma
        # ---------------------------------------------------------
        E_all = []
        O_all = []

        for k in range(cfg["R"]):
            gamma_k = gammas[k]
            samples_k = samples[k]
            logpsi_k = logpsi_vals[k]

            M = samples_k.shape[0]
            E_k = []
            O_k = []

            for j in range(M):
                sigma_j = samples_k[j]
                logpsi_σ = logpsi_k[j]

                # Local energy
                EL_j = compute_local_energy(
                    sigma_j, gamma_k,
                    logpsi_fn, logpsi_σ,
                    nn_edges, nnn_edges,
                    J1=1.0, J2=gamma_k
                )
                E_k.append(EL_j)

                # Log derivative (O-operator)
                O_j = jax.grad(logpsi_fn, argnums=1)(
                    sigma_j, gamma_k
                )
                O_k.append(O_j)

            E_all.append(jnp.array(E_k))
            O_all.append(jnp.array(O_k))

        # ---------------------------------------------------------
        # SR update
        # ---------------------------------------------------------
        G, S = compute_sr_matrices(O_all, E_all)
        delta = sr_update(G, S, cfg["eta"], cfg["diag_shift"])

        # Apply update to params
        params = jax.tree_util.tree_map(
            lambda p, dp: p + dp,
            params, delta
        )

        # ---------------------------------------------------------
        # Log energy
        # ---------------------------------------------------------
        mean_energy = jnp.mean(jnp.concatenate(E_all))
        log["energy"].append(float(mean_energy))

        print(f"[Iter {it:03d}] E = {mean_energy:.6f}   dt = {time.time()-t0:.2f}s")

    # Save log
    with open(os.path.join(cfg["logdir"], "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("Training finished.")

    params_path = os.path.join(cfg["logdir"], "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    print(f"Saved parameters -> {params_path}")
    
    return log


if __name__ == "__main__":
    train_fnqs_j1j2()

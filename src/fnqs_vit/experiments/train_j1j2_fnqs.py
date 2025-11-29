# src/fnqs_vit/experiments/train_j1j2_fnqs.py

import os
import time
import json
import pickle

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from fnqs_vit.fnqs.model import FNQSViT
from fnqs_vit.hamiltonians.lattice import create_square_lattice
from fnqs_vit.vmc.local_energy import compute_local_energy_batch
from fnqs_vit.vmc.sampler_heisenberg import (
    random_spin_state_batch,
    random_spin_state_in_sector,
    sample_chain_batch_edges,
    prepare_edge_array,
)
from fnqs_vit.vmc.sr import compute_sr_matrices, sr_update
from fnqs_vit.utils.device import prefer_gpu_if_available, print_available_devices


def get_config(fast=False):
    base = {"Sztarget": 0}  # None for full space

    if fast:
        base.update({
            "L": 4,
            "patch_size": (2,2),
            "dim": 16,
            "depth": 2,
            "hidden_dim": 64,
            "num_heads": 4,
            "R": 2,
            "gamma_min": 0.4,
            "gamma_max": 0.6,
            "n_discard": 20,
            "samples_per_system": 64,
            "iterations": 10,
            "eta": 0.01,
            "diag_shift": 0.01,
            "seed": 42,
            "logdir": "logs/fnqs_j1j2_fast",
        })
    else:
        base.update({
            "L": 6,
            "patch_size": (2,2),
            "dim": 72,
            "depth": 4,
            "hidden_dim": 288,
            "num_heads": 12,
            "R": 10,
            "gamma_min": 0.4,
            "gamma_max": 0.6,
            "n_discard": 20,
            "samples_per_system": 200,
            "iterations": 50,
            "eta": 0.01,
            "diag_shift": 0.01,
            "seed": 42,
            "logdir": "logs/fnqs_j1j2",
        })

    return base


def make_gamma_field(g, Lx, Ly):
    return jnp.ones((Lx, Ly)) * g


def train_fnqs_j1j2(fast=False):

    cfg = get_config(fast)

    print("\n============== TRAINING CONFIG ==============")
    for k,v in cfg.items():
        print(f"{k:20s} : {v}")
    print("=============================================\n")

    Sztarget = cfg["Sztarget"]

    Lx = Ly = cfg["L"]
    nn_edges, nnn_edges = create_square_lattice(Lx, Ly)

    # create edge array for both sampler & energy
    edge_array = prepare_edge_array(nn_edges, nnn_edges)

    gammas = jnp.linspace(cfg["gamma_min"], cfg["gamma_max"], cfg["R"])
    gamma_fields = jnp.stack([make_gamma_field(g, Lx, Ly) for g in gammas])

    # init model
    key = jax.random.PRNGKey(cfg["seed"])
    model = FNQSViT(
        depth=cfg["depth"],
        embed_dim=cfg["dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        patch_size=cfg["patch_size"],
        gamma_mode="structured",
    )

    sigma0 = jnp.ones((Lx,Ly))
    gamma0 = jnp.ones((Lx,Ly)) * 0.5
    params = model.init(key, sigma0, gamma0)

    flat_params, unravel_fn = ravel_pytree(params)

    def logpsi_fn(p, s, g):
        return model.apply(p, s, g)

    log = {"energy": []}

    # ---------------------------------------------------
    for it in range(cfg["iterations"]):
        t0 = time.time()

        E_all = []
        O_all = []

        for k in range(cfg["R"]):
            g_scalar = float(gammas[k])
            g_field = gamma_fields[k]

            # prepare M initial states
            key, k1 = jax.random.split(key)
            M = cfg["samples_per_system"]

            if Sztarget is None:
                sigma_init = random_spin_state_batch(k1, M, Lx, Ly)
            else:
                sigma_init = random_spin_state_in_sector(k1, M, Lx, Ly, Sztarget)

            # sample
            sigma_hist, logpsi_hist = sample_chain_batch_edges(
                key,
                logpsi_fn,
                params,
                g_field,
                sigma_init,
                cfg["n_discard"],
                M,
                edge_array,
                Lx, Ly,
                Sztarget=Sztarget,
            )

            sigma_batch = sigma_hist[-1]
            logpsi_batch = logpsi_hist[-1]

            # compute local energy
            E_k = compute_local_energy_batch(
                sigma_batch,
                g_scalar,
                logpsi_fn,
                logpsi_batch,
                edge_array,
                J1=1.0,
                J2=g_scalar,
                params=params,
            )

            # O operators
            def real_wave(P, s):
                return jnp.real(logpsi_fn(P, s, g_field))
            def imag_wave(P, s):
                return jnp.imag(logpsi_fn(P, s, g_field))

            O_real_tree = jax.vmap(lambda s: jax.grad(real_wave)(params, s))(sigma_batch)
            O_imag_tree = jax.vmap(lambda s: jax.grad(imag_wave)(params, s))(sigma_batch)

            O_tree = jax.tree.map(lambda r,i: r + 1j*i, O_real_tree, O_imag_tree)

            # flatten
            O_flat = []
            for m in range(M):
                o, _ = ravel_pytree(jax.tree.map(lambda x: x[m], O_tree))
                O_flat.append(o)

            O_mat = jnp.stack(O_flat)

            O_all.append(O_mat)
            E_all.append(E_k)

        # SR step
        G, S = compute_sr_matrices(O_all, E_all)
        delta = sr_update(G, S, cfg["eta"], cfg["diag_shift"])
        delta_tree = unravel_fn(delta)

        params = jax.tree.map(lambda p,dp: p+dp, params, delta_tree)

        # energy log
        E_concat = jnp.concatenate(E_all)
        meanE = float(jnp.real(jnp.mean(E_concat)))
        imagE = float(jnp.imag(jnp.mean(E_concat)))

        if abs(imagE) > 1e-6:
            print(f"⚠️ Warning: Imag component = {imagE:.3e}")

        log["energy"].append(meanE)
        print(f"[Iter {it:03d}] E = {meanE:.6f}   dt = {time.time()-t0:.2f}s")

    # save
    with open(os.path.join(cfg["logdir"], "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    with open(os.path.join(cfg["logdir"], "params.pkl"), "wb") as f:
        pickle.dump(params, f)

    print("Training finished.")
    return log


if __name__ == "__main__":
    prefer_gpu_if_available()
    print_available_devices()
    train_fnqs_j1j2(fast=True)

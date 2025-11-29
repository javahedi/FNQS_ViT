# src/fnqs_vit/utils/device.py
import jax


def print_available_devices():
    print("JAX default backend:", jax.default_backend())
    for dev in jax.devices():
        print("  -", dev)


def prefer_gpu_if_available():
    # This only has effect if called early, before heavy JAX work
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []

    if len(gpus) > 0:
        print("⚙️  Using GPU backend")
        jax.config.update("jax_platform_name", "gpu")
    else:
        print("⚙️  No GPU found, using CPU backend")
        jax.config.update("jax_platform_name", "cpu")

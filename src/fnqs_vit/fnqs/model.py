# src/fnqs_vit/fnqs/model.py

import jax
import jax.numpy as jnp
from flax import linen as nn

from fnqs_vit.fnqs.patching import (
    extract_sigma_patches,
    extract_gamma_patches,
)

from fnqs_vit.fnqs.embedding import MultimodalEmbedding
from fnqs_vit.fnqs.transformer import TransformerEncoder


class FNQSViT(nn.Module):
    """
    Foundation Neural-Network Quantum State based on a Vision Transformer.

    ψθ(σ, γ) = exp[ f_amp(z) + i f_phase(z) ]

    Parameters
    ----------
    depth : int
        Number of transformer layers.
    embed_dim : int
        Dimension of ViT embeddings (d).
    hidden_dim : int
        Hidden size of MLP block in transformer.
    num_heads : int
        Number of attention heads.
    patch_size : tuple
        (px, py) patch size, e.g., (2, 2).
    gamma_mode : str
        "small" or "structured".
    translational_invariant : bool
        If True, use TI-attention.
    """
    depth: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    patch_size: tuple = (2, 2)
    gamma_mode: str = "structured"
    translational_invariant: bool = False

    @nn.compact
    def __call__(self, sigma, gamma):
        """
        Compute logψ(σ, γ).

        Parameters
        ----------
        sigma : jnp.ndarray, shape (Lx, Ly)
        gamma : depends on gamma_mode:
            - "small": shape (C,)
            - "structured": shape (Lx, Ly) or (Lx, Ly, C)

        Returns
        -------
        logpsi : complex jnp.ndarray, shape ()
        """

        # -------------------------------------------------------------
        # 1. Extract patches
        # -------------------------------------------------------------
        sigma_patches = extract_sigma_patches(sigma, self.patch_size)

        if self.gamma_mode == "structured":
            gamma_patches = extract_gamma_patches(gamma, self.patch_size)
        else:
            gamma_patches = gamma   # shape (C,)

        # -------------------------------------------------------------
        # 2. Multimodal embedding
        # -------------------------------------------------------------
        x = MultimodalEmbedding(
            embed_dim=self.embed_dim,
            gamma_mode=self.gamma_mode
        )(sigma_patches, gamma_patches)

        # -------------------------------------------------------------
        # 3. Transformer Encoder
        # -------------------------------------------------------------
        x = TransformerEncoder(
            depth=self.depth,
            dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            translational_invariant=self.translational_invariant
        )(x)

        # -------------------------------------------------------------
        # 4. Global aggregation
        # -------------------------------------------------------------
        z = jnp.sum(x, axis=0)  # shape (embed_dim,)

        # -------------------------------------------------------------
        # 5. Output heads: amplitude + phase
        # -------------------------------------------------------------
        amp = nn.Dense(1)(z).squeeze()     # real scalar
        phs = nn.Dense(1)(z).squeeze()     # real scalar

        return amp + 1j * phs

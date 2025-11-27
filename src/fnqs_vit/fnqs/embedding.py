# src/fnqs_vit/fnqs/embedding.py

import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """
    Linear embedding of flattened patches.

    Parameters
    ----------
    embed_dim : int
        Output dimension of each patch embedding.
    """
    embed_dim: int

    @nn.compact
    def __call__(self, patches):
        # patches: (n_patches, patch_dim)
        return nn.Dense(self.embed_dim)(patches)


class MultimodalEmbedding(nn.Module):
    """
    Multimodal embedding for FNQS-ViT.
    Handles both σ patches and γ patches.

    Two modes:
    ----------
    (A) Small γ (O(1)):
        - γ is a vector of shape (C,)
        - concat γ to each σ-patch before embedding

    (B) Structured γ (O(N)):
        - γ is patch-wise (same shape as σ-patches)
        - embed σ-patches and γ-patches separately
        - concatenate after embedding

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension (d).
    gamma_mode : str
        "small" | "structured"
    """
    embed_dim: int
    gamma_mode: str = "structured"   # default for FNQS

    @nn.compact
    def __call__(self, sigma_patches, gamma_input):
        """
        Parameters
        ----------
        sigma_patches : jnp.ndarray
            Shape (n_patches, px*py)
        gamma_input : jnp.ndarray
            Depending on gamma_mode:
            - "small": shape (C,)
            - "structured": shape (n_patches, px*py*C)

        Returns
        -------
        embeddings : jnp.ndarray
            Shape (n_patches, embed_dim)
        """

        if self.gamma_mode == "small":
            # Case A: concatenate γ to each σ patch
            n_patches = sigma_patches.shape[0]
            gamma_repeated = jnp.repeat(
                gamma_input[None, :], repeats=n_patches, axis=0
            )
            patches = jnp.concatenate([sigma_patches, gamma_repeated], axis=-1)
            return PatchEmbedding(self.embed_dim)(patches)

        # Case B: structured γ (default for J1–J2)
        # embed σ and γ separately
        half = self.embed_dim // 2

        emb_sigma = PatchEmbedding(half)(sigma_patches)
        emb_gamma = PatchEmbedding(half)(gamma_input)

        # combine
        return jnp.concatenate([emb_sigma, emb_gamma], axis=-1)

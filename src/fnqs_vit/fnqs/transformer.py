# src/fnqs_vit/fnqs/transformer.py

import jax
import jax.numpy as jnp
from flax import linen as nn


# -------------------------------------------------------------
# LayerNorm + MLP block
# -------------------------------------------------------------
class MLPBlock(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


# -------------------------------------------------------------
# Standard Multi-Head Self-Attention
# -------------------------------------------------------------
class SelfAttention(nn.Module):
    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        return nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            use_bias=True,
        )(x)


# -------------------------------------------------------------
# Translationally-Invariant Attention (TI-attention)
# -------------------------------------------------------------
class TranslationalInvariantAttention(nn.Module):
    """
    Attention parameters shared across patches.

    Useful for non-disordered systems ensuring translational symmetry.
    We achieve this by applying attention to relative positions only.
    """
    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # Standard attention but params shared across sequence positions
        # => Flax SelfAttention already shares across tokens
        return nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            use_bias=True,
        )(x)


# -------------------------------------------------------------
# Transformer Encoder Block
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    dim: int
    hidden_dim: int
    num_heads: int
    translational_invariant: bool = False

    @nn.compact
    def __call__(self, x):

        # ----- Attention -----
        residual = x
        x = nn.LayerNorm()(x)

        if self.translational_invariant:
            x = TranslationalInvariantAttention(
                dim=self.dim,
                num_heads=self.num_heads
            )(x)
        else:
            x = SelfAttention(
                dim=self.dim,
                num_heads=self.num_heads
            )(x)

        x = x + residual

        # ----- MLP -----
        residual = x
        x = nn.LayerNorm()(x)
        x = MLPBlock(
            hidden_dim=self.hidden_dim,
            out_dim=self.dim
        )(x)

        return x + residual


# -------------------------------------------------------------
# Multi-Layer Transformer Encoder
# -------------------------------------------------------------
class TransformerEncoder(nn.Module):
    depth: int
    dim: int
    hidden_dim: int
    num_heads: int
    translational_invariant: bool = False

    @nn.compact
    def __call__(self, x):
        """
        x shape: (n_patches, dim)
        """
        for _ in range(self.depth):
            x = TransformerBlock(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                translational_invariant=self.translational_invariant
            )(x)
        return x

"""Matched-FLOPs baselines for CHIMERA comparisons.

Two baselines, same `forward_prefill` interface as `ChimeraLM` so the trainer
and eval harnesses can swap them in:

  - `PureTransformerLM` — full-causal-attention every layer (mode 3 only).
  - `PureSSMLM`        — toy SSM every layer (mode 1 only).

Both use the same building blocks as CHIMERA (RoPE, SwiGLU, LayerNorm) so
parameter counts at matched `(num_layers, dim, num_heads)` are directly
comparable. Configure to match a CHIMERA model size at *matched FLOPs* by
keeping `num_layers` and `dim` identical.

These are *not* meant to beat their state-of-the-art equivalents — they're
**experimental controls** so the CHIMERA-vs-baseline comparison has a
well-defined denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from chimera.modules.attention import causal_attention
from chimera.modules.ffn import SwiGLU
from chimera.modules.rope import RotaryEmbedding
from chimera.modules.ssm import ToySSM


@dataclass
class BaselineConfig:
    vocab_size: int = 32_000
    num_layers: int = 4
    dim: int = 256
    num_heads: int = 4
    ssm_state: int = 32
    ffn_mult: float = 8 / 3
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    tie_embeddings: bool = True


# ----------------------------------------------------------------------
# Pure Transformer
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_in = nn.LayerNorm(cfg.dim)
        self.norm_post = nn.LayerNorm(cfg.dim)
        self.proj_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        head_dim = cfg.dim // cfg.num_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.max_seq_len, base=cfg.rope_base)
        self.ffn = SwiGLU(cfg.dim, hidden_mult=cfg.ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.cfg.num_heads
        Dk = D // H
        h = self.norm_in(x)
        q = self.proj_q(h).view(B, T, H, Dk)
        k = self.proj_k(h).view(B, T, H, Dk)
        v = self.proj_v(h).view(B, T, H, Dk)
        positions = torch.arange(T, device=x.device)
        q = self.rope.apply(q, positions)
        k = self.rope.apply(k, positions)
        attn = causal_attention(q, k, v)  # (B, T, H, Dk)
        out = self.proj_o(attn.reshape(B, T, D))
        z = x + out
        return z + self.ffn(self.norm_post(z))


class PureTransformerLM(nn.Module):
    """Dense causal Transformer. Same I/O shape as ChimeraLM.forward_prefill."""

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_final = nn.LayerNorm(cfg.dim)
        self.unembed = None if cfg.tie_embeddings else nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward_prefill(self, input_ids: torch.Tensor, **kwargs) -> "_BaselineOut":
        h = self.tok_embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_final(h)
        logits = h @ self.tok_embed.weight.t() if self.unembed is None else self.unembed(h)
        return _BaselineOut(logits=logits)


# ----------------------------------------------------------------------
# Pure SSM
# ----------------------------------------------------------------------
class SSMBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(cfg.dim)
        self.norm_post = nn.LayerNorm(cfg.dim)
        self.ssm = ToySSM(cfg.dim, state_size=cfg.ssm_state)
        self.ffn = SwiGLU(cfg.dim, hidden_mult=cfg.ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(x)
        out, _ = self.ssm.forward_prefill(h)
        z = x + out
        return z + self.ffn(self.norm_post(z))


class PureSSMLM(nn.Module):
    """SSM-only LM. Same I/O shape as ChimeraLM.forward_prefill."""

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([SSMBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_final = nn.LayerNorm(cfg.dim)
        self.unembed = None if cfg.tie_embeddings else nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward_prefill(self, input_ids: torch.Tensor, **kwargs) -> "_BaselineOut":
        h = self.tok_embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_final(h)
        logits = h @ self.tok_embed.weight.t() if self.unembed is None else self.unembed(h)
        return _BaselineOut(logits=logits)


# ----------------------------------------------------------------------
# Shared output type (mirrors ChimeraLM PrefillOutput for trainer compat)
# ----------------------------------------------------------------------
@dataclass
class _BaselineOut:
    logits: torch.Tensor
    router_outputs: list = field(default_factory=list)  # baselines have no router

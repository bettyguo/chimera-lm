"""ChimeraBlock — the composition unit.

Replaces the standard `attention -> ffn` Transformer block with:

    h     = LN_in(x)
    g     = Router(h)                     # gate distribution over K modes
    o_k   = mixer_k(h)  for k in 0..K-1   # 4 modes (identity, SSM, SWA, full)
    y_mix = sum_k g_k * o_k               # soft or hard one-hot mix
    z     = x + y_mix
    out   = z + FFN(LN_post(z))

Mode dispatch:
  0 (identity): out = h
  1 (SSM):      out = SSM.step / forward_prefill
  2 (SWA):      out = SlidingWindowAttn over ring (always-written under Resolution A)
  3 (full):     out = FullCausalAttn over mode-3-tagged subset (+ self)

For PREFILL: all four modes are computed for all positions; the gate selects.
This is wasteful at hard inference but correct, and keeps the implementation
simple. A Phase-3 optimization batches tokens within mode.

For DECODE: same dispatch, but reading from the cache. The K/V write contract:
  - SSM state is always updated.
  - Ring buffer is always written (Resolution A).
  - Persistent KV is only written when the hard mode == 3.
Writes happen *after* the read for the same step (so the mode-2/3 read includes
the current token).

Causal-consistency contract (tests/test_causal_consistency.py): for any input
and any routing, prefill and step-by-step decode produce identical outputs
to fp64 bit-exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera.cache import ChimeraCacheLayer
from chimera.modules.attention import NEG_INF, sliding_window_attention, step_attention
from chimera.modules.ffn import SwiGLU
from chimera.modules.rope import RotaryEmbedding
from chimera.modules.router import MODE_FULL, MODE_SWA, MixerRouter, RouterOutput
from chimera.modules.ssm import ToySSM


@dataclass
class BlockConfig:
    """Hyperparameters for a single ChimeraBlock."""

    dim: int
    num_heads: int
    window: int = 512
    ssm_state: int = 64
    target_dist: tuple[float, ...] = (0.10, 0.60, 0.20, 0.10)
    ffn_mult: float = 8 / 3
    max_seq_len: int = 8192
    rope_base: float = 10_000.0

    @property
    def head_dim(self) -> int:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by num_heads {self.num_heads}")
        return self.dim // self.num_heads


class ChimeraBlock(nn.Module):
    """A single CHIMERA decoder block.

    Public methods:
      forward_prefill(x, router_mode="soft") -> (y, RouterOutput)
      forward_decode_step(x_t, cache, position, router_mode="hard_top1")
          -> (y_t, RouterOutput)
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.norm_in = nn.LayerNorm(cfg.dim)
        self.norm_post = nn.LayerNorm(cfg.dim)

        self.router = MixerRouter(cfg.dim, target=cfg.target_dist)

        # Mode 1: SSM with the production (step, forward_prefill) interface.
        # Mamba-2 SSD will drop in here on GPU.
        self.ssm = ToySSM(cfg.dim, state_size=cfg.ssm_state)

        # Modes 2 & 3: shared QKV projections (one routing decision per token).
        self.proj_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj_o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.max_seq_len, base=cfg.rope_base
        )

        self.ffn = SwiGLU(cfg.dim, hidden_mult=cfg.ffn_mult)

    # ------------------------------------------------------------------
    # Prefill (whole-sequence)
    # ------------------------------------------------------------------
    def forward_prefill(
        self, x: torch.Tensor, *, router_mode: str = "soft"
    ) -> tuple[torch.Tensor, RouterOutput]:
        """Forward over a full sequence.

        x: (B, T, D)
        returns:
          y:          (B, T, D)
          router_out: RouterOutput  (one_hot/weights have shape (B, T, K))
        """
        B, T, D = x.shape
        H = self.cfg.num_heads
        Dk = self.cfg.head_dim

        h = self.norm_in(x)
        router_out = self.router(h, mode=router_mode)

        # Mode 0: identity in the post-LN space.
        out_0 = h

        # Mode 1: SSM over the whole sequence (always-on under Resolution A).
        out_1, _ = self.ssm.forward_prefill(h)

        # Modes 2 & 3 share Q/K/V.
        q = self.proj_q(h).view(B, T, H, Dk)
        k = self.proj_k(h).view(B, T, H, Dk)
        v = self.proj_v(h).view(B, T, H, Dk)
        positions = torch.arange(T, device=x.device)
        q = self.rope.apply(q, positions)
        k = self.rope.apply(k, positions)

        # Mode 2: sliding-window causal attention (ring is always written, so
        # every prior token is in the lookback window).
        attn_2 = sliding_window_attention(q, k, v, self.cfg.window)
        out_2 = self.proj_o(attn_2.reshape(B, T, D))

        # Mode 3: causal attention restricted to mode-3-tagged positions, with
        # the current position always included. Mask shape (B, T, T).
        out_3 = self._mode3_prefill(q, k, v, router_out.hard_index)

        # Combine via gate. one_hot is the straight-through one-hot (hard) or
        # the soft weights (soft).
        gates = router_out.one_hot.unsqueeze(-1)  # (B, T, K, 1)
        stacked = torch.stack([out_0, out_1, out_2, out_3], dim=2)  # (B, T, K, D)
        mixed = (gates * stacked).sum(dim=2)  # (B, T, D)

        # Pre-LN residual + FFN.
        z = x + mixed
        y = z + self.ffn(self.norm_post(z))
        return y, router_out

    def _mode3_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hard_index: torch.Tensor,
    ) -> torch.Tensor:
        """Mode-3 attention with mask = (j<=i) AND (m_j==3 OR i==j), per batch.

        q, k, v:     (B, T, H, Dk)
        hard_index:  (B, T) — argmax mode per token
        returns:     (B, T, D)
        """
        B, T, H, Dk = q.shape
        D = H * Dk
        s = 1.0 / math.sqrt(Dk)
        qh = q.transpose(1, 2)  # (B, H, T, Dk)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * s  # (B, H, T, T)

        i_idx = torch.arange(T, device=q.device).unsqueeze(1)
        j_idx = torch.arange(T, device=q.device).unsqueeze(0)
        causal = j_idx <= i_idx  # (T, T)
        is_mode3 = (hard_index == MODE_FULL).unsqueeze(1)  # (B, 1, T)
        diag = torch.eye(T, dtype=torch.bool, device=q.device).unsqueeze(0)  # (1, T, T)
        mask = causal.unsqueeze(0) & (is_mode3.expand(B, T, T) | diag)  # (B, T, T)

        scores = scores.masked_fill(~mask.unsqueeze(1), NEG_INF)
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, vh)  # (B, H, T, Dk)
        return self.proj_o(attn.transpose(1, 2).contiguous().view(B, T, D))

    # ------------------------------------------------------------------
    # Decode (single step)
    # ------------------------------------------------------------------
    def forward_decode_step(
        self,
        x_t: torch.Tensor,
        cache: ChimeraCacheLayer,
        position: int,
        *,
        router_mode: str = "hard_top1",
    ) -> tuple[torch.Tensor, RouterOutput]:
        """Single decode step.

        x_t:      (B, D) — current token's input (pre-LN).
        cache:    per-layer cache; mutated in place after the read.
        position: integer position index for RoPE.

        Note on batched decode: this implementation assumes all batch elements
        share the same hard routing decision at this step. Heterogeneous routing
        across the batch requires a per-element dispatch (Phase 3 optimization).
        For the parity test (B=1), this is exact.
        """
        B, D = x_t.shape
        H = self.cfg.num_heads
        Dk = self.cfg.head_dim

        h = self.norm_in(x_t)
        # Run the router on (B, 1, D) so the existing module works.
        router_out = self.router(
            h.unsqueeze(1), mode=router_mode, update_balancer=False
        )

        # Always-on SSM update (mode-1's read uses the post-update state).
        ssm_out, new_ssm_state = self.ssm.step(h, cache.ssm_state)

        # Always-needed K/V for the ring write (Resolution A) and for mode 2/3 reads.
        q_t = self.proj_q(h).view(B, 1, H, Dk)
        k_t = self.proj_k(h).view(B, 1, H, Dk)
        v_t = self.proj_v(h).view(B, 1, H, Dk)
        positions = torch.tensor([position], device=x_t.device, dtype=torch.long)
        q_t = self.rope.apply(q_t, positions).squeeze(1)  # (B, H, Dk)
        k_t = self.rope.apply(k_t, positions).squeeze(1)
        v_t = v_t.squeeze(1)

        # Compute all four mode outputs — needed for soft routing parity. For
        # hard routing only one would suffice; we compute all for code-clarity.
        out_0 = h
        out_1 = ssm_out

        # Mode 2: ring + current (view-with-current).
        rk2, rv2 = cache.view_with_current(k_t, v_t, MODE_SWA)
        attn_2 = step_attention(q_t, rk2, rv2)  # (B, H, Dk)
        out_2 = self.proj_o(attn_2.reshape(B, D))

        # Mode 3: persistent + current.
        rk3, rv3 = cache.view_with_current(k_t, v_t, MODE_FULL)
        attn_3 = step_attention(q_t, rk3, rv3)
        out_3 = self.proj_o(attn_3.reshape(B, D))

        gates = router_out.one_hot.squeeze(1).unsqueeze(-1)  # (B, K, 1)
        stacked = torch.stack([out_0, out_1, out_2, out_3], dim=1)  # (B, K, D)
        mixed = (gates * stacked).sum(dim=1)  # (B, D)

        # Commit writes after the read. Use the hard argmax for the write mode
        # so soft training and hard decode see the same cache contents.
        # For batched decode we'd dispatch per-element; here B=1 in practice.
        write_mode = int(router_out.hard_index[0, 0].item())
        cache.commit_write(k_t, v_t, write_mode, new_ssm_state)

        z = x_t + mixed
        y_t = z + self.ffn(self.norm_post(z))
        return y_t, router_out

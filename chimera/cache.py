"""Multi-mode KV/state cache for CHIMERA.

Implements Resolution A from spec Appendix A.2: every token writes K/V to the
ring buffer and SSM state; only mode-3 tokens write to the persistent KV cache.
The router decides what each query *reads*, not what each prior token wrote.

The decode contract this module enforces (with `tests/test_causal_consistency.py`
as the canary):

  for any sequence x_1..x_T and any routing pattern g_1..g_T,
    prefill(x).logits  ≡  [decode_step(x_t, g_t, cache) for t in 1..T]
  to bit-exact precision in fp64 (< 1e-10) and < 1e-4 in bf16.

Bug class to watch: off-by-one in the ring/persistent view at decode time. The
read for token t must include (k_t, v_t) — write happens *after* the read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ChimeraCacheLayer:
    """Per-layer multi-mode cache state.

    Shapes (with K, V channels treated as the head-flattened tensor (B, H * Dk)
    or per-head (B, H, Dk); both supported as long as the attention impl agrees):

      ssm_state          (B, S)
      ring_k, ring_v     (B, W, *kv_dim)
      ring_size          int — current entries in the ring (cap W)
      persistent_k/_v    list of (B, *kv_dim), length = #mode-3 tokens written

    Attributes:
      window:   sliding window size W
      d_ssm:    SSM state size S
    """

    window: int
    d_ssm: int
    ssm_state: torch.Tensor
    ring_k: torch.Tensor
    ring_v: torch.Tensor
    ring_size: int = 0
    persistent_k: list[torch.Tensor] = field(default_factory=list)
    persistent_v: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        batch: int,
        kv_shape: tuple[int, ...],
        d_ssm: int,
        window: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ChimeraCacheLayer:
        """Construct an empty cache.

        kv_shape is the per-token K/V shape *without* the batch dim, e.g.
        (H, Dk) for multi-head or (D,) for flat.
        """
        return cls(
            window=window,
            d_ssm=d_ssm,
            ssm_state=torch.zeros(batch, d_ssm, device=device, dtype=dtype),
            ring_k=torch.zeros(batch, window, *kv_shape, device=device, dtype=dtype),
            ring_v=torch.zeros(batch, window, *kv_shape, device=device, dtype=dtype),
        )

    def commit_write(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        mode: int,
        new_ssm_state: torch.Tensor,
    ) -> None:
        """Write the current step's K/V into the appropriate buffers.

        Must be called *after* the read for the same step has happened. The
        read should have used the "view with current token" — see
        :meth:`view_with_current`.
        """
        self.ssm_state = new_ssm_state
        # Ring is always written under Resolution A.
        if self.ring_size < self.window:
            self.ring_k[:, self.ring_size] = k_t
            self.ring_v[:, self.ring_size] = v_t
            self.ring_size += 1
        else:
            # Shift left by one, append at the end.
            self.ring_k = torch.cat([self.ring_k[:, 1:], k_t.unsqueeze(1)], dim=1)
            self.ring_v = torch.cat([self.ring_v[:, 1:], v_t.unsqueeze(1)], dim=1)
        if mode == 3:
            self.persistent_k.append(k_t)
            self.persistent_v.append(v_t)

    def view_with_current(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K/V window the query at step t should attend over.

        For mode 2: ring contents + current (k_t, v_t), capped at window W.
        For mode 3: persistent cache + current.
        For modes 0/1: undefined (callers should not invoke this).
        """
        if mode == 2:
            if self.ring_size < self.window:
                rk = torch.cat([self.ring_k[:, : self.ring_size], k_t.unsqueeze(1)], dim=1)
                rv = torch.cat([self.ring_v[:, : self.ring_size], v_t.unsqueeze(1)], dim=1)
            else:
                rk = torch.cat([self.ring_k[:, 1:], k_t.unsqueeze(1)], dim=1)
                rv = torch.cat([self.ring_v[:, 1:], v_t.unsqueeze(1)], dim=1)
            return rk, rv
        if mode == 3:
            pk = torch.stack([*self.persistent_k, k_t], dim=1)
            pv = torch.stack([*self.persistent_v, v_t], dim=1)
            return pk, pv
        raise ValueError(f"view_with_current undefined for mode {mode}")


@dataclass
class ChimeraCache:
    """Container for per-layer caches.

    Held externally by the inference loop / trainer — never as an `nn.Module`
    attribute, so multiple independent generation streams don't trample each
    other's state.
    """

    layers: list[ChimeraCacheLayer]

    @classmethod
    def empty(
        cls,
        num_layers: int,
        batch: int,
        kv_shape: tuple[int, ...],
        d_ssm: int,
        window: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ChimeraCache:
        return cls(
            layers=[
                ChimeraCacheLayer.empty(
                    batch, kv_shape, d_ssm, window, device=device, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> ChimeraCacheLayer:
        return self.layers[idx]

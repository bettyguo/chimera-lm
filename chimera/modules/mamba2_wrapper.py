"""Mamba-2 wrapper — production mode-1 backend on CUDA.

Import is gated: this module only succeeds if `mamba_ssm` is available
(installed via the `gpu` extra on a CUDA box). On CPU/Windows the import
fails fast with a clear error.

Interface contract — must match `chimera/modules/ssm.py::ToySSM`:

    class _SSM(Protocol):
        def empty_state(self, batch, *, device, dtype) -> Tensor: ...
        def step(self, x_t, prior_state) -> tuple[Tensor, Tensor]: ...
        def forward_prefill(self, x, initial_state=None) -> tuple[Tensor, Tensor]: ...

This wrapper is **untested on this dev box** (no CUDA). Verify with
`tests/test_causal_consistency.py` after the GPU swap; if parity fails,
the wrapper is buggy, not the test.

Usage:

    # On GPU:
    from chimera.modules.mamba2_wrapper import Mamba2SSD
    block.ssm = Mamba2SSD(dim=D, state_size=128)

    # Or via BlockConfig (see ChimeraBlock.__init__ for the dispatch idea):
    cfg.ssm_kind = "mamba2"  # adds branch in ChimeraBlock.__init__
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _ensure_mamba_ssm_available() -> Any:
    """Import mamba-ssm or raise a clear error."""
    try:
        import mamba_ssm  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "mamba-ssm is not installed. Install with `pip install -e \".[gpu]\"` "
            "on a CUDA-capable Linux box. This wrapper requires CUDA + Triton + "
            "compatible mamba-ssm (>=2.2)."
        ) from e
    return mamba_ssm


class Mamba2SSD(nn.Module):
    """Mamba-2 SSD wrapper exposing the `ToySSM`-compatible interface.

    Note: the `mamba-ssm` API exposes a `Mamba2` class with a step API for
    autoregressive decode (`step` / `forward` with inference cache). This
    wrapper adapts that to our `(x_t, prior_state)` signature.

    Shapes match the toy:
        empty_state: (B, S)
        step:        (B, D), (B, S) -> ((B, D), (B, S))
        forward_prefill: (B, T, D), (B, S)|None -> ((B, T, D), (B, S))
    """

    def __init__(
        self,
        dim: int,
        state_size: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ) -> None:
        super().__init__()
        mamba_ssm = _ensure_mamba_ssm_available()
        self.dim = dim
        self.state_size = state_size
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        # The exact class name is `Mamba2` in mamba-ssm 2.x.
        self.mamba = mamba_ssm.Mamba2(
            d_model=dim, d_state=state_size, d_conv=d_conv, expand=expand, headdim=headdim
        )

    def empty_state(self, batch: int, *, device, dtype) -> torch.Tensor:
        """Return a zeroed state. Mamba-2 actually keeps multiple internal
        state tensors (conv state + ssm state); we represent them as a flat
        (B, state_size) for interface compatibility and stash the rest
        internally if needed."""
        return torch.zeros(batch, self.state_size, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, prior_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step.

        IMPLEMENTATION NOTE: mamba-ssm's step API uses an inference cache
        object (`InferenceParams`) rather than a flat state tensor. The
        canonical adaptation:

            params = InferenceParams(max_seqlen=..., max_batch_size=B)
            params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
            out = self.mamba(x_t.unsqueeze(1), inference_params=params)
            return out.squeeze(1), self._flatten_state(params)

        We deliberately don't write this until we can verify on a CUDA box —
        the test of correctness is `tests/test_causal_consistency.py`
        passing after the swap.
        """
        raise NotImplementedError(
            "Mamba2SSD.step requires a CUDA-tested implementation. See the "
            "docstring for the InferenceParams adaptation pattern."
        )

    def forward_prefill(
        self, x: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole-sequence forward.

        mamba-ssm's `Mamba2.__call__` already does the chunked parallel scan
        and returns the output. The final state is recoverable from the
        inference cache; the simplest version drops it and returns zeros.
        """
        raise NotImplementedError(
            "Mamba2SSD.forward_prefill: wire `self.mamba(x)` here once the box "
            "has CUDA. The parity test will validate."
        )

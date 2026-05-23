# ADR-002 — Toy SSM in v1; Mamba-2 swap deferred to GPU

## Status

Accepted (Phase 1). Supersedable once the GPU box is ready.

## Context

The CHIMERA spec specifies Mamba-2 SSD (`mamba-ssm` package) as the mode-1
mixer. That package:

- Requires CUDA + nvcc + a working Triton install.
- Does not build on Windows (the dev box for this scaffold).
- Cannot run on CPU at all (kernels are CUDA-only).

We need *some* mode-1 mixer to test the multi-mode block: the cache contract
is meaningless without a function attached.

## Decision

Implement `ToySSM` in `chimera/modules/ssm.py` — a first-order linear
recurrence with sigmoid decay — exposing the same interface Mamba-2 will
satisfy:

```python
class SSMModule(Protocol):
    def step(self, x_t: Tensor, prior_state: Tensor) -> tuple[Tensor, Tensor]:
        """One decode step: (B, D), (B, S) -> ((B, D), (B, S))."""
    def forward_prefill(self, x: Tensor, initial_state: Tensor | None = None
                        ) -> tuple[Tensor, Tensor]:
        """Whole sequence: (B, T, D) -> ((B, T, D), (B, S))."""
    def empty_state(self, batch: int, *, device, dtype) -> Tensor: ...
```

`ToySSM` is a *not* the production model — it has no selective gating, no
data-dependent decay, and the state dimension is small. But its outputs are
*differentiable*, *causal*, and *deterministic*, which is all the cache
contract requires.

## Consequences

**Positive**

- All tests run on the CPU dev box. `tests/test_causal_consistency.py`
  passes at fp64 bit-exactness.
- The swap to Mamba-2 is a one-file replacement: write a `MambaSSDWrapper`
  satisfying the same interface, and `ChimeraBlock.__init__` chooses by
  config.

**Negative**

- LM quality of the toy SSM is *bad* — first-order recurrence forgets
  fast. Don't quote any perplexity numbers off a model with `ToySSM`.
- We can't validate that the routing distribution under realistic training
  ends up looking like the target `(0.10, 0.60, 0.20, 0.10)` because the
  SSM is too weak to be the default; the router will likely move mass
  toward attention even at small scale. That validation lands when
  Mamba-2 is in.

## Phase-1.5 migration

```python
# chimera/modules/ssm.py
from mamba_ssm import Mamba2

class Mamba2SSD(nn.Module):
    """Production SSM wrapper. Same interface as ToySSM."""
    def __init__(self, dim, state_size=128, ...):
        ...
        self.mamba = Mamba2(d_model=dim, d_state=state_size, ...)
    def step(self, x_t, prior_state):
        # Mamba2 has a step API for autoregressive decode; wrap it.
        ...
    def forward_prefill(self, x, initial_state=None):
        # Mamba2's __call__ does the chunked scan; just call it.
        ...
```

In `BlockConfig` add an `ssm_kind: Literal["toy", "mamba2"] = "toy"` and
dispatch in `ChimeraBlock.__init__`.

`tests/test_causal_consistency.py` should *also* pass with Mamba-2 because
the parity contract is interface-level. If it doesn't, the wrapper has a
bug — fix the wrapper, don't relax the test.

## Same pattern applies to attention

Modes 2 and 3 use pure-PyTorch scaled-dot-product with explicit masks
(`chimera/modules/attention.py`). The swap to flash-attn 2.6's `block_mask`
variant is the same shape: same function signature, different backend.
Defer to ADR-003 if/when that swap is needed.

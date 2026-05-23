# Dao & Gu 2024 — Mamba-2: Transformers are SSMs (State Space Duality)

> **Status:** TODO — unread.

## Primitive operation
TODO — SSD (state-space duality) formulation: a structured-attention view of
selective SSMs enabling chunked parallel scans.

## Complexity
- Prefill (chunked scan): TODO (paper claims wall-clock parity with attention at small T)
- Decode per token: TODO
- State memory: TODO

## Failure mode the paper itself admits
TODO — likely the same recall failure as Mamba-1; state size limits long-range
exact lookup.

## Empirically strongest result
TODO

## Relevance to CHIMERA
**Production mode-1 mixer.** `chimera/modules/ssm.py::ToySSM` shims out the
SSD until the GPU swap. The `step(x_t, state) → (out, new_state)` interface
is modeled directly on Mamba-2's decode API. See ADR-002.

## Cited by
- `chimera/modules/ssm.py`
- `docs/decisions/ADR-002-toy-ssm-substitution.md`

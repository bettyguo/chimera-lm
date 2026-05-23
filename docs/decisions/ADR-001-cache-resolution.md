# ADR-001 — Cache Resolution A for v1

## Status

Accepted (Phase 1).

## Context

Spec Appendix A.2 presents two valid resolutions for the multi-mode cache:

- **Resolution A**: every token writes K/V to the ring buffer; only mode-3
  tokens write to persistent. A mode-3 query sees the mode-3-tagged subset
  ∪ self; a mode-2 query sees the (fully populated) sliding window ∪ self.
- **Resolution B**: mode-1 tokens compress into the SSM state and are
  opaque to subsequent mode-2/3 queries. A mode-3 query reads persistent
  ∪ {learned SSM summary} ∪ self. Achieves the full ~10× KV memory
  reduction but requires a learned "SSM-to-attention bridge" module.

## Decision

Use Resolution A for v1 (Phases 1–4). Defer Resolution B to v2 as an
explicit ablation.

## Consequences

**Positive**

- Provably correct without a learned bridge. The prefill-decode parity
  invariant is straightforward to verify (`tests/test_causal_consistency.py`).
- Implementation is `view_with_current` + write-after-read. ~50 LoC.
- Empirical KV-memory reduction at the headline (T=32K, target dist) is
  still ~8.7× over dense — close to the 10× ideal.

**Negative**

- Mode-2 query sees prior mode-1-routed tokens via the ring (would be the
  same in Resolution B *for mode 2*). Fine.
- Mode-3 query *does not* see prior mode-1-routed tokens. This is the same
  in Resolution A and B for mode 3 specifically. In B, the bridge module
  partially compensates; in A, we accept the loss.
- KV-memory headline is 8.7× (A) instead of 10× (B's theoretical limit
  assuming the bridge is free).

## Phase-1.5 / v2 migration path

Resolution B is a swap of two pieces:

1. `view_with_current(mode=3)` would call into the new bridge module:
   ```python
   summary = self.ssm_to_attn_bridge(ssm_state)   # learned
   return persistent_with_current + summary       # concatenated as extra KV
   ```
2. The mode-3 prefill mask would mask out non-mode-3 positions (already
   does), but the bridge module needs to be trained alongside.

Until the bridge is trained, Resolution B will *hurt* quality compared to
A. Don't enable it without an ablation run.

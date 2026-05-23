# Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces

> **Status:** TODO — unread. Fill in while reading. Do not commit fabricated content.

## Primitive operation
TODO — selective state-space recurrence with input-dependent A, B, C matrices.

## Complexity
- Prefill / train: TODO
- Decode per token: TODO
- State memory at step T: TODO

## Failure mode the paper itself admits
TODO

## Empirically strongest result
TODO

## Relevance to CHIMERA
TODO — Mamba is the *predecessor* of Mamba-2 (next note). Mamba-2's SSD interface
is the production mode-1 mixer; this paper establishes selective gating that
makes the SSM data-dependent.

## Cited by
- `chimera/modules/ssm.py` (interface only — Mamba-2 swap-in)

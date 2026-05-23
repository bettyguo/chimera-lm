# Waleffe et al. 2024 — Empirical Study of Mamba-Hybrid Scaling Laws

> **Status:** TODO — unread.

## Primitive operation
TODO — empirical scaling-law fits for Mamba and Mamba-hybrid stacks.

## Complexity
N/A — empirical paper.

## Failure mode the paper itself admits
TODO — likely: small fraction of attention layers (~5–15%) is necessary to
recover competitive perplexity; pure Mamba lags slightly.

## Empirically strongest result
TODO — the headline scaling-law plot: loss vs. compute, with fitted
exponents for pure-Transformer, pure-Mamba, and hybrid.

## Relevance to CHIMERA
**Sets the bar for the scaling-law comparison.** Spec §4 Phase 4: "Fit a
scaling law `L = E + A/N^α + B/D^β`. Compare exponents vs. baselines."
CHIMERA must show *competitive or better* exponents to claim the
hybridization-ratio choice matters at scale.

## Cited by
- `scripts/scaling_laws.py` (planned)

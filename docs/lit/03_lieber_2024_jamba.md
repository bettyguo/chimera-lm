# Lieber et al. 2024 — Jamba: A Hybrid Transformer-Mamba Language Model

> **Status:** TODO — unread.

## Primitive operation
TODO — interleaved Transformer + Mamba blocks at a *fixed* ratio across the
stack. Plus MoE FFN in some configurations.

## Complexity
TODO — dominated by the attention layers; SSM layers reduce average cost.

## Failure mode the paper itself admits
TODO — hybridization ratio is hand-tuned per the depth; not data-dependent.

## Empirically strongest result
TODO

## Relevance to CHIMERA
**The baseline CHIMERA generalizes.** Jamba's hybrid ratio is fixed at design
time; CHIMERA's is learned per-token per-layer via the router. Phase 4
ablation: `ablations/fixed_ratio_baseline.py` reproduces a Jamba-style ratio
at matched FLOPs.

## Cited by
- `ablations/fixed_ratio_baseline.py` (planned)

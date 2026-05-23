# Arora et al. 2024 — Zoology / Based: The Recall Problem in Linear Models

> **Status:** TODO — unread.

## Primitive operation
TODO — diagnostic suite (MQAR = Multi-Query Associative Recall) and proposed
fix (Based: gated linear attention + light convolution).

## Complexity
TODO

## Failure mode the paper itself admits
TODO — but the *whole point* of the paper is to *expose* the recall failure
mode of linear models. Quote the headline scaling claim: linear-model recall
accuracy drops as T grows, attention is invariant.

## Empirically strongest result
TODO — the MQAR plots at T ∈ {64, 128, ..., 4K} showing the gap between
attention and SSM.

## Relevance to CHIMERA
**The recall benchmark CHIMERA must win.** Spec §4 Phase 4 exit criterion:
"CHIMERA-large beats Mamba-2 baseline on MQAR by ≥20 percentage points at
T ≥ 4K." Our synthetic MQAR generator in `chimera/data/synthetic.py` is a
faithful (smaller-scale) version of this paper's protocol.

The CHIMERA thesis: the router will learn to send the *query positions* in
the MQAR sequence to mode 3 (full attention), recovering exact recall.

## Cited by
- `chimera/data/synthetic.py::make_mqar_batch`
- `eval/mqar.py` (planned)

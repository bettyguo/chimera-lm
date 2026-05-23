# Glorioso et al. 2024 — Zamba; Ren et al. 2024 — Samba

> **Status:** TODO — both unread.

## Zamba

### Primitive operation
TODO — Mamba backbone with a single shared attention block applied periodically.

### Complexity
TODO

### Failure mode admitted
TODO

### Strongest result
TODO

## Samba

### Primitive operation
TODO — interleaved Mamba + sliding-window attention (no full attention).

### Complexity
TODO

### Failure mode admitted
TODO

### Strongest result
TODO

## Relevance to CHIMERA

Both are *fixed-ratio* hybrid baselines like Jamba. Samba is particularly
relevant because it pairs SSM with **sliding-window** attention (CHIMERA's
mode 2), with no full attention (mode 3). A natural ablation: CHIMERA with
target dist `(0.10, 0.60, 0.30, 0.00)` — i.e., disable mode 3 — and compare
against Samba at matched FLOPs.

## Cited by
- `ablations/fixed_ratio_baseline.py` (planned)

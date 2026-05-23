# Raposo et al. 2024 — Mixture-of-Depths

> **Status:** TODO — unread.

## Primitive operation
TODO — per-token routing through *layer skipping*: each token decides whether
to execute the current layer or pass through untouched.

## Complexity
TODO — saves compute proportional to skip rate; routing is top-k over tokens
per layer.

## Failure mode the paper itself admits
TODO

## Empirically strongest result
TODO

## Relevance to CHIMERA
**Closest prior art to CHIMERA's router.** MoD routes between {execute layer,
skip layer}; CHIMERA routes between {identity, SSM, SWA, full}. CHIMERA's
"identity" mode (mode 0) is literally MoD's "skip" choice. The straight-through
hard-routing scheme in `chimera/modules/router.py` follows MoD's design.

## Cited by
- `chimera/modules/router.py`
- `docs/routing.md`

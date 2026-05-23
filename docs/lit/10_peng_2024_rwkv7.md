# Peng et al. 2024 — RWKV-7

> **Status:** TODO — unread.

## Primitive operation
TODO — data-dependent state evolution: the recurrence parameters are
input-dependent, like Mamba's selective scan but with a different
parameterization (channel-wise time decay, token-dependent receptance).

## Complexity
- Decode per token: TODO (O(1) state-update)
- Prefill: TODO

## Failure mode the paper itself admits
TODO — same family-level limitation as Mamba: bounded state size means
bounded exact recall.

## Empirically strongest result
TODO — perplexity numbers vs. Mamba-2 and Transformer at matched compute.

## Relevance to CHIMERA
**Family reference, not implemented.** RWKV-7 is a *competitor* to Mamba-2
in the data-dependent-recurrence space. We do not use RWKV-7 as mode 1 in
v1 because Mamba-2 has more mature CUDA kernels and a cleaner step interface.

If a future ablation wants to swap mode 1 from Mamba-2 to RWKV-7, the
`(step, forward_prefill)` interface in `chimera/modules/ssm.py` is the
extension point.

## Cited by
- `chimera/modules/ssm.py` (alternative backend, not yet implemented)

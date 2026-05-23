# Multi-mode KV / state cache

## What it stores

`ChimeraCacheLayer` (`chimera/cache.py`) holds, per layer:

| Buffer | Shape | Write rule | Read rule |
|---|---|---|---|
| `ssm_state` | (B, S) | always | mode-1 reads post-update state |
| `ring_k`, `ring_v` | (B, W, H, Dk) | always (Resolution A) | mode-2 reads window ∪ self |
| `ring_size` | int | tracks fill (capped at W) | — |
| `persistent_k`, `persistent_v` | list of (B, H, Dk) | only when hard mode == 3 | mode-3 reads list ∪ self |

`ChimeraCache` is a thin container over a list of per-layer caches.

## Resolution A vs Resolution B

The spec gives two valid resolutions for "how does a mode-3 query see prior
mode-1-routed tokens?" We use **Resolution A** in v1:

- *Resolution A:* every token writes its (k, v) to the ring buffer regardless
  of route. Only mode-3 tokens additionally write to persistent. This means
  a mode-3 query sees the mode-3-tagged subset (plus itself), and a mode-2
  query sees all prior tokens within window (plus itself). It loses some of
  the theoretical 10× memory advantage but is provably correct without a
  learned "SSM-to-attention bridge".

- *Resolution B (v2):* mode-1 tokens are opaque to subsequent mode-2/3 queries.
  Persistent contains only mode-3 tokens (same as A). A mode-3 query reads
  persistent **plus a learned summary of the SSM state** at that position.
  This achieves the full memory reduction but requires training a bridge
  module.

ADR-001 records this choice. The architectural delta is contained in
`ChimeraCacheLayer.view_with_current` and the mode-3 prefill mask.

## The read-write order at decode

The critical correctness condition at decode step t:

> The mode-2/3 read at step t must include `(k_t, v_t)` (the current token).
> So: build the view-with-current first, attend, then commit the write.

Wrong order:

```python
cache.commit_write(k_t, v_t, mode, new_ssm)  # WRONG — ring shifted by one
out = step_attention(q_t, cache.ring_k, cache.ring_v)
```

Right order:

```python
view_k, view_v = cache.view_with_current(k_t, v_t, mode)
out = step_attention(q_t, view_k, view_v)
cache.commit_write(k_t, v_t, mode, new_ssm)   # commit AFTER read
```

`ChimeraCacheLayer.view_with_current(k_t, v_t, mode)` returns the appropriate
window:

- mode 2: ring (capped at W) ∪ `{(k_t, v_t)}`.
- mode 3: persistent list ∪ `{(k_t, v_t)}`.

For modes 0 and 1, `view_with_current` raises — callers handle those cases
directly (mode 0 returns `h`; mode 1 returns the post-update SSM output).

## Ring eviction

When `T > W`, the ring is full and new writes evict the oldest entry. The
implementation is `torch.cat([ring[:, 1:], k_t[None]], dim=1)` (shift left,
append at end). The reference impl uses `torch.cat` for clarity; an in-place
shift would be faster but easy to get subtly wrong. `test_block_parity_ring_eviction`
verifies eviction matches prefill's sliding-window mask.

## Per-layer KV memory at step T

With routing fractions `(f_0, f_1, f_2, f_3)`:

```
mem(layer) = bytes_per_elem · (S + 2·D·W + 2·D·f_3·T)
           = (SSM state) + (ring) + (persistent)
```

At `T = 32K`, `D = 2048`, `W = 512`, `S = 128`, `f_3 = 0.10`, `bf16` (2 bytes):

```
SSM state:   2 · 128                = 0.26 KB
Ring:        2 · 2 · 2048 · 512     = 4.0 MB
Persistent:  2 · 2 · 2048 · 3276.8  = ~26.8 MB
Total:                                ~30.8 MB
```

Dense Transformer per-layer KV at `T = 32K`:

```
2 · 2 · 2048 · 32768 = ~268 MB
```

→ **~8.7× smaller** at this routing distribution. Matches the spec headline.

Verified by `tests/test_profiling.py::test_headline_decode_memory_chimera_smaller_than_dense`.

## Parity contract (the canary)

`tests/test_causal_consistency.py` asserts:

```
max_t |prefill_logits[t] - decode_logits[t]|_∞ < 1e-9    (fp64)
```

For every: single-mode routing, mixed routing, ring-eviction-stress, and
full-model end-to-end. Run this on every PR. If it fails: write a
POSTMORTEM; do not silently fix the off-by-one.

Current measured diffs:

| Test | T | W | Diff |
|---|---|---|---|
| block, hard | 8 / 32 / 128 | 8 | ~9e-16 |
| block, soft | 8 / 32 / 128 | 8 | ~9e-16 |
| block, ring eviction | 64 | 8 | ~9e-16 |
| model (2 layers), hard | 16 | 8 | ~1e-14 |
| model (2 layers), soft | 16 | 8 | ~1e-14 |
| model, ring eviction | 24 | 4 | ~1e-14 |

Per-block diff is at machine epsilon; the model-level diff is the per-block
epsilon × number of layers × multiplicative noise in the final norm + unembed.
Both are bit-exact for our purposes.

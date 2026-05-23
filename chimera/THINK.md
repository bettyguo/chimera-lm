# Phase 1 THINK.md — Single ChimeraBlock + end-to-end model

## 1. What I understood the task to be

Build, test, and forward-pass-verify a single ChimeraBlock that composes four
sequence-mixing modes (identity, SSM, sliding-window attention, full causal
attention) under a learned per-token router with aux-loss-free load balancing.
Compose blocks into `ChimeraLM`. Provide a multi-mode KV cache and prove
prefill-decode equivalence to fp64 bit-exactness via the canary test in
`tests/test_causal_consistency.py`.

End-state for Phase 1:

- `chimera/modules/router.py` — `MixerRouter` + `AuxFreeBalancer` (spec A.1).
- `chimera/modules/ssm.py` — SSM with `(step, forward_prefill)` interface.
- `chimera/modules/attention.py` — causal full + sliding-window attention.
- `chimera/modules/{ffn,rope}.py` — SwiGLU + RoPE.
- `chimera/modules/chimera_block.py` — composed block, Resolution A (spec A.2).
- `chimera/cache.py` — multi-mode cache with `view_with_current` + `commit_write`.
- `chimera/model.py` — end-to-end `ChimeraLM` with prefill / decode / generate.
- `chimera/losses.py`, `chimera/utils/profiling.py` — diagnostics + A.4 formulas.
- Test suite passing in fp64 bit-exact.

## 2. Alternative designs considered

**Design A: implement everything in pure PyTorch on CPU first.** Selected.
- Mamba-2 SSD and flash-attn need CUDA + nvcc; they don't build on the target
  workstation (`torch 2.9.1+cpu`, Windows). We need *some* SSM and *some*
  attention to test the cache contract. We substitute toys whose interfaces
  match production, so the GPU swap is a wiring change.
- Pros: every test runs on the dev machine; no CUDA debugging in Phase 1.
- Cons: training-scale runs aren't possible here; that's deferred to a
  GPU box.

**Design B: shim out the SSM and attention as no-ops; only test the cache
bookkeeping.** Rejected. The cache contract is meaningless without a real
function attached to each mode — silent miscompiles like RoPE-position drift
slip through. We want the actual numerical parity to hold.

**Design C: use HuggingFace transformers as the attention backend.** Rejected.
Drags in tokenizer / config plumbing we don't need; we'd then have to rip it
back out for the flash-attn swap. A 50-line scaled-dot-product is sufficient.

**Design D: compute only the selected mode in prefill (hard routing).**
Rejected for Phase 1. It complicates the dataflow (gather/scatter or per-mode
batching) and obscures the parity test. We compute all four mode outputs at
every position in prefill and combine via the router's one-hot or soft
weights. This wastes prefill FLOPs in hard mode but is obviously correct.
A Phase 3 optimization batches tokens by mode.

## 3. Chosen design & explicit tradeoffs

**Block dataflow (Pre-LN):**

```
x ── LN_in ── h
     │
     ├── Router → (weights, hard_index, one_hot)
     │
     ├── identity:    out_0 = h
     ├── SSM:         out_1 = SSM.forward_prefill(h)
     ├── SWA:         out_2 = SWA(qkv(h)) over window
     └── full causal: out_3 = full_attn(qkv(h)) over mode-3 ∪ self
     │
     mixed = Σ one_hot_k * out_k         # soft or hard
     z = x + mixed                        # residual
     out = z + FFN(LN_post(z))            # post-block
```

**Cache contract (Resolution A from spec A.2):**

- SSM state is always updated.
- Ring buffer is always written (every token).
- Persistent KV is only written when the hard argmax mode == 3.

The router decides what each *query* reads, not what each prior token wrote.
A mode-3 query at position i reads `{persistent ∪ {(k_i, v_i)}}` — i.e., the
mode-3-tagged history plus itself. A mode-2 query reads the ring (window of
all prior tokens) plus self.

**Tradeoffs accepted:**

- *Toy SSM.* `ToySSM` is a first-order linear recurrence with sigmoid decay,
  not the selective state-space of Mamba-2. The *interface* matches Mamba-2's
  `step(x_t, state) → (out, new_state)`. Production swap = one file change.
  → ADR-002.
- *Pure-PyTorch attention.* Standard scaled-dot-product with explicit masks,
  not flash-attn's block_mask. Same I/O signature. Production swap = one
  file change. → ADR-003 (deferred).
- *Compute-all-modes prefill.* Hard routing in prefill still computes all
  four mode outputs (we just gate to one-hot). Wastes prefill FLOPs but
  preserves the parity invariant straightforwardly. → ADR-004 (deferred).
- *B=1 hard decode.* `forward_decode_step` assumes all batch elements share
  the same hard mode at this step (it dispatches once). Heterogeneous batch
  routing is a Phase 3 optimization.

## 4. What could go wrong (failure modes, ranked)

1. **Off-by-one in the ring/persistent view at decode.** The mode-2/3 read
   for step t must include `(k_t, v_t)`. The cache's `view_with_current`
   exists precisely so the read happens *before* the write. Caught by
   `test_block_parity_hard`, `test_block_parity_ring_eviction`.
2. **RoPE phase drift across prefill vs decode.** Prefill applies RoPE at
   positions `0..T-1` in one shot; decode applies it at `t` per step. Both
   must use the same precomputed table and the same position arg. Caught by
   the parity tests.
3. **Soft routing mode-3 read undefined at non-mode-3 positions.** Defined
   as "attend over mode-3-tagged set ∪ {self}". This is what decode also
   sees (persistent_k + [k_t]), so parity holds. Caught by
   `test_block_parity_soft`.
4. **Balancer mutation between prefill and decode.** If the router updates
   the balancer mid-step, decode at step t sees a different bias than
   prefill at position t. Mitigation: `update_balancer` is gated by
   `self.training`. Parity tests run in `.eval()`. Caught by
   `test_balancer_no_update_in_eval`.
5. **Cache ring eviction off by one when `T > W`.** Eviction shifts left
   and appends at the end. Caught by `test_block_parity_ring_eviction` and
   `test_cache_ring_eviction_keeps_window_size`.
6. **Mode-3 mask edge: position 0 with no prior mode-3 history.** Mask is
   `(j<=i) AND (m_j==3 OR i==j)`. At i=0, only j=0 is allowed; softmax over
   one position gives weight 1, output is v_0. Decode sees persistent=[] +
   [k_0], same single-element softmax. Caught by parity tests.
7. **Batched decode with heterogeneous routing.** Currently dispatches once
   on `hard_index[0, 0]`. For B>1 with diverging routing, this is wrong.
   Documented; parity tests use B=1.

## 5. Evidence-of-success (what makes Phase 1 done)

- [x] `pytest tests/` passes 30/30 in fp64.
- [x] `test_block_parity_{hard,soft}` at T ∈ {8, 32, 128}: diff < 1e-15 per block.
- [x] `test_block_parity_ring_eviction` at T=64, W=8: diff < 1e-15.
- [x] `test_model_parity_{hard,soft}` at 2 layers, T=16: diff < 1.5e-14.
- [x] Soft-routing forward+backward at (B=2, T=16, D=32) — gradient flows
      through router, SSM, attention, FFN (`test_block_gradient_flow_soft`).
- [x] FLOP/memory accounting (A.4) matches the headline numbers within
      rounding (`test_headline_decode_memory_chimera_smaller_than_dense`).

## 6. Out of scope (Phase 2+)

- Mamba-2 SSD wiring; flash-attn wiring (Phase 1.5 on GPU).
- Trainer (`train/train.py`), streaming dataloader, FineWeb-Edu setup (Phase 3).
- Heterogeneous batched decode (Phase 3 optimization).
- Pre-allocated persistent KV growth (Phase 2.5).
- Eval harness: MQAR, needle-in-haystack, LongBench (Phase 4).
- Ablations (Phase 5).

## 7. References

- Spec `01_CHIMERA.md` Appendix A.1 — aux-free balancer math.
- Spec `01_CHIMERA.md` Appendix A.2 — Resolution A; the parity contract.
- Spec `01_CHIMERA.md` Appendix A.4 — FLOP / memory accounting formulas.
- Spec `01_CHIMERA.md` Appendix B — reference cache impl (lifted into
  `chimera/cache.py` and `tests/test_causal_consistency.py`).

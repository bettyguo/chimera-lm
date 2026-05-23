"""Synthetic datasets for CHIMERA — CPU-runnable diagnostics.

Two tasks:

  - **Multi-Query Associative Recall (MQAR)** — the headline benchmark from
    Arora et al. 2024 "Zoology": the model sees a sequence of (key, value)
    pairs and then queries for values by key. SSMs collapse here; attention
    nails it. The CHIMERA thesis says the router learns to route the query
    positions to mode-3 (full attention).

  - **Selective copy** — a simpler diagnostic. The model sees a sequence of
    "noise" tokens with a few "remember-me" markers, then a "reproduce them"
    cue. Tests whether the architecture can find specific positions.

Both tasks are sequence-level next-token prediction: the loss is computed only
on the answer positions (others are `-100` so cross-entropy ignores them).

Token layout for MQAR (vocab = num_keys + num_vals + num_special):

    [ k1, v1, k2, v2, ..., kN, vN, QUERY_TOKEN, q1, ?, q2, ?, ..., qM, ? ]

The `?` slots are where the model must produce the correct vi for each qi.
We mask out non-`?` positions so the LM loss only cares about the answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SPECIAL_PAD = 0
SPECIAL_QUERY = 1
NUM_SPECIAL = 2


@dataclass
class MQARConfig:
    num_keys: int = 32                # |K|
    num_values: int = 32              # |V|
    num_kv_pairs: int = 8             # N: pairs stored before the query block
    num_queries: int = 4              # M: queries to answer
    answer_slot_token: int = 0        # input value at `?` slots; ignored by loss anyway

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL + self.num_keys + self.num_values

    @property
    def key_range(self) -> tuple[int, int]:
        return NUM_SPECIAL, NUM_SPECIAL + self.num_keys

    @property
    def value_range(self) -> tuple[int, int]:
        return (
            NUM_SPECIAL + self.num_keys,
            NUM_SPECIAL + self.num_keys + self.num_values,
        )

    @property
    def seq_len(self) -> int:
        # 2*N (kv pairs) + 1 (QUERY token) + 2*M (qi, ?) - 1 (last ? is the final position,
        # but we still need to predict it, so the model gets shown q_M and predicts ?).
        return 2 * self.num_kv_pairs + 1 + 2 * self.num_queries


def make_mqar_batch(
    cfg: MQARConfig, batch_size: int, *, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of MQAR examples.

    Returns:
      input_ids: (B, T) long — the input sequence.
      targets:   (B, T) long — next-token targets. Positions where loss should
                 NOT be counted are filled with -100. Only the `?` positions
                 (i.e., positions where the *next* token is the answer) have
                 valid targets.

    Convention: cross_entropy expects logits at position t to predict token
    at position t+1. So the answer for question q_i (at position p) must be
    at input position p+1 (the `?` slot), and the target at position p must
    be the value v_i.

    To make this straightforward, we fill the answer slot with a placeholder
    in the input and put the correct value in `targets[p]` where p is the
    position of q_i.
    """
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    B = batch_size
    N = cfg.num_kv_pairs
    M = cfg.num_queries
    T = cfg.seq_len

    key_lo, key_hi = cfg.key_range
    val_lo, val_hi = cfg.value_range

    # Sample N distinct keys per batch element (no replacement) — required for
    # the lookup to be unambiguous. Use torch.argsort over uniform noise.
    if N > cfg.num_keys:
        raise ValueError(f"num_kv_pairs={N} > num_keys={cfg.num_keys}")
    if M > N:
        raise ValueError(f"num_queries={M} > num_kv_pairs={N} (cannot query unknown keys)")

    # Distinct key sampling per batch element.
    noise = torch.rand(B, cfg.num_keys, generator=g)
    perm = noise.argsort(dim=-1)[:, :N] + key_lo  # (B, N) — N distinct keys per row

    # Values: sampled iid (different keys can share values is fine for MQAR).
    values = torch.randint(val_lo, val_hi, (B, N), generator=g)

    # Assemble the kv block: [k1, v1, k2, v2, ...]
    kv_block = torch.stack([perm, values], dim=-1).view(B, 2 * N)  # (B, 2N)

    # Query block: pick M of the N pairs to query (without replacement).
    q_idx = torch.rand(B, N, generator=g).argsort(dim=-1)[:, :M]  # (B, M) indices into kv pairs
    q_keys = perm.gather(1, q_idx)        # (B, M)
    q_values = values.gather(1, q_idx)    # (B, M) — the ground truth

    # Interleave queries with placeholder answer slots: [q1, ?, q2, ?, ...]
    placeholders = torch.full((B, M), cfg.answer_slot_token, dtype=torch.long)
    q_block = torch.stack([q_keys, placeholders], dim=-1).view(B, 2 * M)  # (B, 2M)

    # QUERY separator
    query_token = torch.full((B, 1), SPECIAL_QUERY, dtype=torch.long)

    input_ids = torch.cat([kv_block, query_token, q_block], dim=1)  # (B, T)
    assert input_ids.shape == (B, T), f"expected (B, T)=(B, {T}), got {tuple(input_ids.shape)}"

    # Targets: -100 except at positions where the *next* token should be a value.
    # The query positions in the sequence are at indices [2N+1, 2N+3, 2N+5, ..., 2N+2M-1].
    # The model sees q_i at position (2N + 1 + 2i); the next-token target there is v_i.
    targets = torch.full((B, T), -100, dtype=torch.long)
    base = 2 * N + 1
    for i in range(M):
        pos = base + 2 * i           # position of q_i
        targets[:, pos] = q_values[:, i]

    return input_ids, targets


@dataclass
class CopyConfig:
    """Selective-copy diagnostic.

    Layout: [noise_1, ..., noise_N, MARKER, target_1, ..., target_K, CUE, ?, ?, ..., ?]
    The model must reproduce the target tokens at the `?` slots.
    """

    vocab_size: int = 64
    noise_len: int = 16
    target_len: int = 4
    marker_token: int = 0
    cue_token: int = 1
    pad_token: int = 0

    @property
    def seq_len(self) -> int:
        return self.noise_len + 1 + self.target_len + 1 + self.target_len  # +marker +cue +ans


def make_copy_batch(
    cfg: CopyConfig, batch_size: int, *, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective-copy task. Returns (input_ids, targets) with -100 mask."""
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    B = batch_size
    noise = torch.randint(2, cfg.vocab_size, (B, cfg.noise_len), generator=g)
    targets_seq = torch.randint(2, cfg.vocab_size, (B, cfg.target_len), generator=g)
    marker = torch.full((B, 1), cfg.marker_token, dtype=torch.long)
    cue = torch.full((B, 1), cfg.cue_token, dtype=torch.long)
    answer_slots = torch.full((B, cfg.target_len), cfg.pad_token, dtype=torch.long)

    input_ids = torch.cat([noise, marker, targets_seq, cue, answer_slots], dim=1)

    # The model sees the cue at position (noise_len + 1 + target_len), then the
    # first answer slot. Targets for the answer slots are targets_seq.
    targets = torch.full_like(input_ids, -100)
    # Position of CUE = noise_len + target_len + 1.
    # The first answer slot is at CUE+1. Target there should be targets_seq[0],
    # i.e. logits at CUE predict targets_seq[0]. Then the next answer slot
    # predicts targets_seq[1], etc.
    cue_pos = cfg.noise_len + cfg.target_len + 1
    for i in range(cfg.target_len):
        targets[:, cue_pos + i] = targets_seq[:, i]

    return input_ids, targets

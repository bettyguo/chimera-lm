"""Tests for the synthetic MQAR / Copy generators."""

import torch

from chimera.data.synthetic import (
    CopyConfig,
    MQARConfig,
    SPECIAL_QUERY,
    make_copy_batch,
    make_mqar_batch,
)


def test_mqar_shapes():
    cfg = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=4, num_queries=2)
    ids, targets = make_mqar_batch(cfg, batch_size=3, seed=0)
    T = cfg.seq_len
    assert ids.shape == (3, T)
    assert targets.shape == (3, T)
    # The QUERY token sits at index 2*N.
    N = cfg.num_kv_pairs
    assert (ids[:, 2 * N] == SPECIAL_QUERY).all()


def test_mqar_targets_only_at_query_positions():
    cfg = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=6, num_queries=3)
    ids, targets = make_mqar_batch(cfg, batch_size=4, seed=42)
    # Non-target positions must be -100.
    valid_mask = targets != -100
    # Exactly num_queries valid targets per row.
    assert valid_mask.sum(dim=1).eq(cfg.num_queries).all().item()
    # All valid targets are within the value range.
    vlo, vhi = cfg.value_range
    valid_targets = targets[valid_mask]
    assert (valid_targets >= vlo).all().item()
    assert (valid_targets < vhi).all().item()


def test_mqar_targets_match_keys():
    """Sanity: for each query position, find the matching key in the kv block
    and verify the target equals the corresponding value."""
    cfg = MQARConfig(num_keys=16, num_values=16, num_kv_pairs=6, num_queries=3)
    ids, targets = make_mqar_batch(cfg, batch_size=4, seed=99)
    N, M = cfg.num_kv_pairs, cfg.num_queries
    for b in range(4):
        # KV block: positions 0..2N-1
        kv_keys = ids[b, 0:2 * N:2]
        kv_vals = ids[b, 1:2 * N:2]
        base = 2 * N + 1
        for i in range(M):
            q_pos = base + 2 * i
            q_key = ids[b, q_pos]
            assert q_key.item() != cfg.answer_slot_token, "query key must not equal placeholder"
            # Find the kv pair with this key.
            match_idx = (kv_keys == q_key).nonzero(as_tuple=True)[0]
            assert match_idx.numel() == 1, f"key {q_key} must appear exactly once in kv block"
            expected_val = kv_vals[match_idx[0]]
            assert targets[b, q_pos].item() == expected_val.item()


def test_copy_shapes_and_mask():
    cfg = CopyConfig(vocab_size=32, noise_len=8, target_len=3)
    ids, targets = make_copy_batch(cfg, batch_size=3, seed=0)
    T = cfg.seq_len
    assert ids.shape == (3, T) and targets.shape == (3, T)
    # Exactly target_len valid (non-masked) positions per row.
    valid = (targets != -100).sum(dim=1)
    assert valid.eq(cfg.target_len).all().item()


def test_mqar_no_duplicate_keys_per_row():
    cfg = MQARConfig(num_keys=8, num_values=8, num_kv_pairs=8, num_queries=4)
    ids, _ = make_mqar_batch(cfg, batch_size=4, seed=7)
    N = cfg.num_kv_pairs
    keys = ids[:, 0:2 * N:2]  # (B, N)
    for b in range(keys.shape[0]):
        assert keys[b].unique().numel() == N

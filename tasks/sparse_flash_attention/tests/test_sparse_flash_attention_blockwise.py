#!/usr/bin/env python3
"""Focused block-wise correctness probes for SparseFlashAttention.

This file intentionally keeps an independent reference for sparse_block_size > 1
instead of reusing the token-wise reference in test_sparse_flash_attention.py.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import test_sparse_flash_attention as base


def blockwise_reference(case: base.Case) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    bsz, qs, qn, _ = case.query.shape
    kvs = case.key.shape[1]
    block_size = int(case.sparse_block_size)
    output = np.zeros(case.query.shape, dtype=np.float64)
    max_output = np.full((bsz, 1, qs, qn), base.EMPTY_MAX, dtype=np.float64)
    sum_output = np.zeros((bsz, 1, qs, qn), dtype=np.float64)
    scale = float(np.float16(case.scale))

    for b in range(bsz):
        query_len = qs if case.actual_query is None else min(max(int(case.actual_query[b]), 0), qs)
        kv_len = kvs if case.actual_kv is None else min(max(int(case.actual_kv[b]), 0), kvs)
        for q in range(query_len):
            block_indices = case.sparse_indices[b, q, 0].astype(np.int64)
            positions: list[int] = []
            for block_index in block_indices:
                if block_index < 0:
                    continue
                block_start = int(block_index) * block_size
                if block_start >= kv_len:
                    continue
                for block_offset in range(block_size):
                    key_pos = block_start + block_offset
                    if key_pos >= kv_len:
                        break
                    if case.sparse_mode == 3 and key_pos > q + kv_len - query_len:
                        continue
                    positions.append(key_pos)

            if not positions:
                continue

            selected = np.asarray(positions, dtype=np.int64)
            selected_key = case.key[b, selected, 0].astype(np.float64)
            selected_rope = case.key_rope[b, selected, 0].astype(np.float64)
            selected_value = case.value[b, selected, 0].astype(np.float64)
            query = case.query[b, q].astype(np.float64)
            query_rope = case.query_rope[b, q].astype(np.float64)
            scores = (query @ selected_key.T + query_rope @ selected_rope.T) * scale
            row_max = scores.max(axis=1, keepdims=True)
            exponentials = np.exp(scores - row_max)
            row_sum = exponentials.sum(axis=1, keepdims=True)
            output[b, q] = (exponentials / row_sum) @ selected_value
            max_output[b, 0, q] = row_max[:, 0]
            sum_output[b, 0, q] = row_sum[:, 0]

    output = output.astype(case.query.dtype)
    if not case.return_aux:
        return output, None, None
    return output, max_output.astype(np.float32), sum_output.astype(np.float32)


def build_cases() -> list[base.Case]:
    fp16 = np.dtype(np.float16)
    basic = replace(
        base.make_case(
            "block2_basic",
            [[0, 2], [1, 2]],
            kvs=6,
            dtype=fp16,
            sparse_mode=0,
            return_aux=True,
        ),
        sparse_block_size=2,
    )
    causal_tail = replace(
        base.make_case(
            "block2_causal_tail",
            [[0, 2], [1, 2], [0, 2]],
            kvs=6,
            dtype=fp16,
            actual_query=3,
            actual_kv=5,
            sparse_mode=3,
            return_aux=True,
        ),
        sparse_block_size=2,
    )
    return [basic, causal_tail]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()

    runtime = base.AclRuntime(args.device)
    try:
        op = base.SparseFlashAttentionAclnn(runtime)
        failures = 0
        for case in build_cases():
            expected_attention, expected_max, expected_sum = blockwise_reference(case)
            actual_attention, actual_max, actual_sum = op.run(case)

            attention_abs, attention_rel = base.error_metrics(actual_attention, expected_attention)
            max_abs = base.error_metrics(actual_max, expected_max)[0] if actual_max is not None else 0.0
            sum_abs = base.error_metrics(actual_sum, expected_sum)[0] if actual_sum is not None else 0.0
            finite = np.isfinite(actual_attention).all()
            if actual_max is not None:
                finite = finite and np.isfinite(actual_max).all() and np.isfinite(actual_sum).all()

            passed = (
                np.allclose(actual_attention, expected_attention, rtol=2.0e-3, atol=2.0e-3)
                and np.allclose(actual_max, expected_max, rtol=1.0e-4, atol=1.0e-4)
                and np.allclose(actual_sum, expected_sum, rtol=1.0e-4, atol=1.0e-4)
                and finite
            )
            print(
                f"BLOCKWISE_CASE {case.name} "
                f"attention_abs={attention_abs:.9g} attention_rel={attention_rel:.9g} "
                f"max_abs={max_abs:.9g} sum_abs={sum_abs:.9g} "
                f"finite={finite} result={'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            failures += 0 if passed else 1

        print(f"BLOCKWISE_SUMMARY passed={2 - failures} total=2", flush=True)
        return 0 if failures == 0 else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

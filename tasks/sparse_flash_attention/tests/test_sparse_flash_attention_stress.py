#!/usr/bin/env python3
"""True-launch stress coverage for the SparseFlashAttention competition domain."""

from __future__ import annotations

import argparse
from dataclasses import replace
import time

import numpy as np

import test_sparse_flash_attention as base
from test_sparse_flash_attention_blockwise import blockwise_reference


def make_case(
    name: str,
    *,
    bsz: int,
    qs: int,
    qn: int,
    kvs: int,
    sparse_block_size: int,
    sparse_indices: np.ndarray,
    dtype: np.dtype = np.dtype(np.float16),
    sparse_mode: int = 0,
    return_aux: bool = True,
    actual_query: np.ndarray | None = None,
    actual_kv: np.ndarray | None = None,
) -> base.Case:
    """Build bounded deterministic inputs without shape-specific golden data."""
    content_axis = np.arange(512, dtype=np.float32)
    rope_axis = np.arange(64, dtype=np.float32)
    query = np.empty((bsz, qs, qn, 512), dtype=np.float32)
    query_rope = np.empty((bsz, qs, qn, 64), dtype=np.float32)
    key = np.empty((bsz, kvs, 1, 512), dtype=np.float32)
    value = np.empty((bsz, kvs, 1, 512), dtype=np.float32)
    key_rope = np.empty((bsz, kvs, 1, 64), dtype=np.float32)

    query_base = ((content_axis % 23.0) - 11.0) / 256.0
    key_base = ((content_axis % 19.0) - 9.0) / 256.0
    value_base = ((content_axis % 29.0) - 14.0) / 64.0
    query_rope_base = ((rope_axis % 13.0) - 6.0) / 96.0
    key_rope_base = ((rope_axis % 11.0) - 5.0) / 96.0
    for batch in range(bsz):
        for query_pos in range(qs):
            for head in range(qn):
                row_tag = batch * 17 + query_pos * 5 + head
                query[batch, query_pos, head] = (
                    query_base * (1.0 + (row_tag % 7) / 16.0)
                    + ((row_tag % 5) - 2.0) / 512.0
                )
                query_rope[batch, query_pos, head] = (
                    query_rope_base * (1.0 + (row_tag % 9) / 32.0)
                )
        for key_pos in range(kvs):
            key_tag = batch * 13 + key_pos
            key[batch, key_pos, 0] = (
                key_base * (1.0 + (key_tag % 11) / 32.0)
                + ((key_tag % 7) - 3.0) / 1024.0
            )
            value[batch, key_pos, 0] = (
                value_base * (1.0 + (key_tag % 5) / 32.0)
                + ((key_tag % 17) - 8.0) / 128.0
            )
            key_rope[batch, key_pos, 0] = (
                key_rope_base * (1.0 + (key_tag % 13) / 32.0)
            )

    indices = np.asarray(sparse_indices, dtype=np.int32)
    if indices.ndim == 1:
        indices = np.broadcast_to(indices, (bsz, qs, 1, indices.size))
    elif indices.ndim == 2:
        indices = np.broadcast_to(indices[:, :, None, :], (bsz, qs, 1, indices.shape[-1]))
    return base.Case(
        name=name,
        query=np.ascontiguousarray(query, dtype=dtype),
        key=np.ascontiguousarray(key, dtype=dtype),
        value=np.ascontiguousarray(value, dtype=dtype),
        sparse_indices=np.ascontiguousarray(indices, dtype=np.int32),
        actual_query=None if actual_query is None else np.ascontiguousarray(actual_query, dtype=np.int32),
        actual_kv=None if actual_kv is None else np.ascontiguousarray(actual_kv, dtype=np.int32),
        query_rope=np.ascontiguousarray(query_rope, dtype=dtype),
        key_rope=np.ascontiguousarray(key_rope, dtype=dtype),
        scale=0.125,
        sparse_mode=sparse_mode,
        return_aux=return_aux,
        sparse_block_size=sparse_block_size,
    )


def build_cases() -> list[base.Case]:
    cases: list[base.Case] = []
    for block_size in (1, 2, 4, 8, 16, 32, 64, 128):
        kvs = block_size * 3 - 1
        cases.append(
            make_case(
                f"block_{block_size}",
                bsz=1,
                qs=3,
                qn=2,
                kvs=kvs,
                sparse_block_size=block_size,
                sparse_indices=np.asarray([0, 1, 2, -1], dtype=np.int32),
                sparse_mode=3 if block_size % 8 == 0 else 0,
                actual_query=np.asarray([3], dtype=np.int32),
                actual_kv=np.asarray([kvs - 1], dtype=np.int32),
            )
        )

    cases.extend(
        [
            make_case(
                "qn_128_qs_1",
                bsz=1,
                qs=1,
                qn=128,
                kvs=32,
                sparse_block_size=1,
                sparse_indices=np.arange(32, dtype=np.int32),
                return_aux=False,
            ),
            make_case(
                "batch_2_qs_3",
                bsz=2,
                qs=3,
                qn=8,
                kvs=16,
                sparse_block_size=2,
                sparse_indices=np.asarray([0, 2, 4, 6, -1], dtype=np.int32),
                actual_query=np.asarray([3, 2], dtype=np.int32),
                actual_kv=np.asarray([16, 13], dtype=np.int32),
                sparse_mode=3,
            ),
            make_case(
                "batch_4_fp32",
                bsz=4,
                qs=2,
                qn=4,
                kvs=17,
                sparse_block_size=4,
                sparse_indices=np.asarray([0, 1, 3, 4], dtype=np.int32),
                dtype=np.dtype(np.float32),
                return_aux=False,
            ),
        ]
    )
    for sparse_size in (256, 1024, 4096):
        cases.append(
            make_case(
                f"long_sparse_{sparse_size}",
                bsz=1,
                qs=1,
                qn=1,
                kvs=sparse_size,
                sparse_block_size=1,
                sparse_indices=np.arange(sparse_size, dtype=np.int32),
                return_aux=sparse_size != 4096,
            )
        )
    return cases


def compare_case(op: base.SparseFlashAttentionAclnn, case: base.Case) -> bool:
    expected_attention, expected_max, expected_sum = blockwise_reference(case)
    start = time.perf_counter()
    actual_attention, actual_max, actual_sum = op.run(case)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    attention_abs, attention_rel = base.error_metrics(actual_attention, expected_attention)
    attention_ok = np.allclose(actual_attention, expected_attention, rtol=3.0e-3, atol=3.0e-3)
    aux_ok = True
    max_abs = 0.0
    sum_abs = 0.0
    if actual_max is not None:
        max_abs = base.error_metrics(actual_max, expected_max)[0]
        sum_abs = base.error_metrics(actual_sum, expected_sum)[0]
        aux_ok = np.allclose(actual_max, expected_max, rtol=3.0e-4, atol=3.0e-4)
        aux_ok &= np.allclose(actual_sum, expected_sum, rtol=3.0e-4, atol=3.0e-4)
    finite = np.isfinite(actual_attention).all()
    if actual_max is not None:
        finite = finite and np.isfinite(actual_max).all() and np.isfinite(actual_sum).all()
    passed = bool(attention_ok and aux_ok and finite)
    print(
        f"STRESS_CASE {case.name} attention_abs={attention_abs:.9g} "
        f"attention_rel={attention_rel:.9g} max_abs={max_abs:.9g} sum_abs={sum_abs:.9g} "
        f"device_ms={op.last_device_ms:.6f} end_to_end_ms={elapsed_ms:.3f} "
        f"result={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def multicore_consistency(op: base.SparseFlashAttentionAclnn) -> bool:
    single = make_case(
        "single_core_reference",
        bsz=1,
        qs=1,
        qn=1,
        kvs=12,
        sparse_block_size=4,
        sparse_indices=np.asarray([0, 1, 2, -1], dtype=np.int32),
    )
    multi = make_case(
        "multi_core_replica",
        bsz=2,
        qs=2,
        qn=16,
        kvs=12,
        sparse_block_size=4,
        sparse_indices=np.asarray([0, 1, 2, -1], dtype=np.int32),
    )
    multi = replace(
        multi,
        query=np.ascontiguousarray(np.broadcast_to(single.query[0, 0, 0], multi.query.shape)),
        key=np.ascontiguousarray(np.broadcast_to(single.key[0], multi.key.shape)),
        value=np.ascontiguousarray(np.broadcast_to(single.value[0], multi.value.shape)),
        query_rope=np.ascontiguousarray(
            np.broadcast_to(single.query_rope[0, 0, 0], multi.query_rope.shape)
        ),
        key_rope=np.ascontiguousarray(np.broadcast_to(single.key_rope[0], multi.key_rope.shape)),
    )
    single_attention, single_max, single_sum = op.run(single)
    multi_attention, multi_max, multi_sum = op.run(multi)
    expected_attention = np.broadcast_to(single_attention[0, 0, 0], multi_attention.shape)
    expected_max = np.full(multi_max.shape, single_max[0, 0, 0, 0], dtype=np.float32)
    expected_sum = np.full(multi_sum.shape, single_sum[0, 0, 0, 0], dtype=np.float32)
    passed = bool(
        np.array_equal(multi_attention, expected_attention)
        and np.array_equal(multi_max, expected_max)
        and np.array_equal(multi_sum, expected_sum)
    )
    print(f"MULTICORE_CONSISTENCY result={'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    runtime = base.AclRuntime(args.device)
    try:
        op = base.SparseFlashAttentionAclnn(runtime)
        cases = build_cases()
        passed = sum(compare_case(op, case) for case in cases)
        consistency = multicore_consistency(op)
        total = len(cases) + 1
        passed += int(consistency)
        print(f"STRESS_SUMMARY passed={passed} total={total}", flush=True)
        return 0 if passed == total else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

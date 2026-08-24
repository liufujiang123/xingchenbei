#!/usr/bin/env python3
"""Guarded torch_npu reference for the competition SparseFlashAttention contract.

The public torch_npu API and some released CPU golden code have disagreed about
whether attention_mode=2 aggregates the independent value tensor or the first
512 elements of key.  This runner therefore refuses to act as a reference until
the selected runtime passes two direct K/V fingerprints in the same process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

# Avoid importing torch_npu once through PyTorch backend auto-loading and again
# explicitly below.  The latter registers all required NPU operators.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu

import test_sparse_flash_attention as base


def _torch_dtype(case: base.Case) -> torch.dtype:
    if case.primary_acl_dtype == base.ACL_BF16:
        return torch.bfloat16
    if case.query.dtype == np.float16:
        return torch.float16
    if case.query.dtype == np.float32:
        return torch.float32
    raise TypeError(f"unsupported primary dtype: {case.query.dtype}")


def _cpu_tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    source = np.ascontiguousarray(value)
    if dtype == torch.bfloat16:
        if source.dtype != np.uint16:
            source = base.float32_to_bf16_storage(source.astype(np.float32))
        # Reinterpret the exact competition BF16 storage instead of rounding it
        # a second time through a Python float conversion.
        return torch.from_numpy(source.copy()).view(torch.bfloat16)
    return torch.from_numpy(source).to(dtype=dtype)


def _to_npu(value: np.ndarray | None, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(value)) if dtype is None else _cpu_tensor(value, dtype)
    return tensor.contiguous().npu()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy().copy()


class TorchNpuSparseFlashAttentionReference:
    """Thin wrapper that preserves the competition argument semantics."""

    def __init__(self, device: int) -> None:
        self.device = device
        torch_npu.npu.set_device(device)

    def run(self, case: base.Case) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, bool]:
        primary_dtype = _torch_dtype(case)
        query = _to_npu(case.query, primary_dtype)
        key = _to_npu(case.key, primary_dtype)
        value = _to_npu(case.value, primary_dtype)
        sparse_indices = _to_npu(case.sparse_indices)
        actual_query = _to_npu(case.actual_query)
        actual_kv = _to_npu(case.actual_kv)
        query_rope = _to_npu(case.query_rope, primary_dtype)
        key_rope = _to_npu(case.key_rope, primary_dtype)

        key_before = _to_numpy(key)
        value_before = _to_numpy(value)
        attention, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
            query,
            key,
            value,
            sparse_indices,
            float(case.scale),
            actual_seq_lengths_query=actual_query,
            actual_seq_lengths_kv=actual_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=case.sparse_block_size,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=case.sparse_mode,
            attention_mode=2,
            return_softmax_lse=case.return_aux,
        )
        torch_npu.npu.synchronize()

        inputs_unchanged = np.array_equal(key_before, _to_numpy(key)) and np.array_equal(
            value_before, _to_numpy(value)
        )
        max_output = None if not case.return_aux else _to_numpy(softmax_max)
        sum_output = None if not case.return_aux else _to_numpy(softmax_sum)
        return _to_numpy(attention), max_output, sum_output, inputs_unchanged


def _fingerprint_case(name: str, key_value: float, value_value: float) -> base.Case:
    query = np.zeros((1, 1, 1, 512), dtype=np.float16)
    key = np.full((1, 4, 1, 512), key_value, dtype=np.float16)
    value = np.full((1, 4, 1, 512), value_value, dtype=np.float16)
    return base.Case(
        name=name,
        query=query,
        key=key,
        value=value,
        sparse_indices=np.asarray([[[[3]]]], dtype=np.int32),
        actual_query=np.asarray([1], dtype=np.int32),
        actual_kv=np.asarray([4], dtype=np.int32),
        query_rope=np.zeros((1, 1, 1, 64), dtype=np.float16),
        key_rope=np.zeros((1, 4, 1, 64), dtype=np.float16),
        scale=0.0884,
        sparse_mode=0,
        return_aux=True,
    )


def classify_output_source(op: TorchNpuSparseFlashAttentionReference) -> dict[str, Any]:
    probes = (("kv_7_9", 7.0, 9.0), ("kv_13_29", 13.0, 29.0))
    results = []
    for name, key_value, value_value in probes:
        output, _, _, inputs_unchanged = op.run(_fingerprint_case(name, key_value, value_value))
        first = float(output.reshape(-1)[0])
        results.append(
            {
                "name": name,
                "key": key_value,
                "value": value_value,
                "output_first": first,
                "all_equal_key": bool(np.allclose(output, key_value, rtol=0.0, atol=0.0)),
                "all_equal_value": bool(np.allclose(output, value_value, rtol=0.0, atol=0.0)),
                "inputs_unchanged": inputs_unchanged,
            }
        )

    all_unchanged = all(item["inputs_unchanged"] for item in results)
    if all_unchanged and all(item["all_equal_value"] for item in results):
        source = "VALUE"
    elif all_unchanged and all(item["all_equal_key"] for item in results):
        source = "KEY"
    elif not all_unchanged:
        source = "INPUT_MUTATED"
    else:
        source = "OTHER"
    return {"output_source": source, "probes": results}


def compare_case(op: TorchNpuSparseFlashAttentionReference, case: base.Case) -> dict[str, Any]:
    if case.query.dtype == np.float32 and case.primary_acl_dtype != base.ACL_BF16:
        raise ValueError(
            "the installed torch_npu SparseFlashAttention contract supports FP16/BF16, not FP32"
        )
    expected_attention, expected_max, expected_sum = base.cpu_reference(case)
    actual_attention, actual_max, actual_sum, inputs_unchanged = op.run(case)

    expected_attention_f32 = (
        base.bf16_storage_to_float32(expected_attention)
        if case.primary_acl_dtype == base.ACL_BF16
        else expected_attention.astype(np.float32)
    )
    result: dict[str, Any] = {
        "case": case.name,
        "inputs_unchanged": inputs_unchanged,
        "attention_max_abs_error": float(
            np.max(np.abs(actual_attention.astype(np.float64) - expected_attention_f32.astype(np.float64)))
        ),
    }
    if case.return_aux:
        result["softmax_max_max_abs_error"] = float(
            np.max(np.abs(actual_max.astype(np.float64) - expected_max.astype(np.float64)))
        )
        result["softmax_sum_max_abs_error"] = float(
            np.max(np.abs(actual_sum.astype(np.float64) - expected_sum.astype(np.float64)))
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="all", help="base case name, all, or fingerprint")
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("SPARSE_FLASH_ATTENTION_DEVICE_ID", "0")),
    )
    args = parser.parse_args()

    op = TorchNpuSparseFlashAttentionReference(args.device)
    fingerprint = classify_output_source(op)
    print(json.dumps({"torch_npu_fingerprint": fingerprint}, sort_keys=True), flush=True)
    if args.case == "fingerprint":
        return 0 if fingerprint["output_source"] == "VALUE" else 3
    if fingerprint["output_source"] != "VALUE":
        print(
            "REFUSED: torch_npu runtime did not prove independent VALUE aggregation",
            file=sys.stderr,
            flush=True,
        )
        return 3

    cases = base.build_cases()
    selected = cases if args.case == "all" else [case for case in cases if case.name == args.case]
    if not selected:
        raise SystemExit(f"unknown case: {args.case}")
    for case in selected:
        if case.query.dtype == np.float32 and case.primary_acl_dtype != base.ACL_BF16:
            print(
                json.dumps(
                    {
                        "case": case.name,
                        "status": "SKIP",
                        "reason": "torch_npu SparseFlashAttention does not support FP32",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        print(json.dumps(compare_case(op, case), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

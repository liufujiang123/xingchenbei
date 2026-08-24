#!/usr/bin/env python3
"""ACL-event benchmark for representative SparseFlashAttention regimes."""

from __future__ import annotations

import argparse
import statistics

import test_sparse_flash_attention as base
from test_sparse_flash_attention_stress import build_cases


BENCHMARK_CASES = ("qn_128_qs_1", "long_sparse_256", "long_sparse_1024", "long_sparse_4096")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    cases = {case.name: case for case in build_cases()}
    runtime = base.AclRuntime(args.device)
    try:
        op = base.SparseFlashAttentionAclnn(runtime)
        for name in BENCHMARK_CASES:
            case = cases[name]
            op.run(case)  # compilation/runtime warm-up, excluded from samples
            samples = []
            for _ in range(args.repeats):
                op.run(case)
                samples.append(op.last_device_ms)
            print(
                f"BENCH label={args.label} case={name} repeats={args.repeats} "
                f"device_ms_median={statistics.median(samples):.6f} "
                f"device_ms_min={min(samples):.6f} device_ms_max={max(samples):.6f}",
                flush=True,
            )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark the retained MhcSinkhorn ACLNN path with CANN runtime events."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
from pathlib import Path

import numpy as np

from test_mhc_sinkhorn_acl import AclRuntime, DeviceArray, MhcSinkhornAclnn


WARMUP = 5
ACTIVE = 9


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if "perf" not in case.get("tags", []):
            continue
        shape = tuple(int(value) for value in case["shape"])
        n = shape[-1]
        if len(shape) < 2 or shape[-2] != n or n not in (4, 6, 8):
            raise ValueError(f"line {line_number}: invalid [..., N, N] shape")
        matrix_count = math.prod(shape[:-2]) if len(shape) > 2 else 1
        cases.append(
            {
                **case,
                "shape": shape,
                "matrix_count": matrix_count,
                "n": n,
            }
        )
    return cases


class EventTimer:
    def __init__(self, runtime: AclRuntime) -> None:
        self.runtime = runtime
        lib = runtime.lib
        lib.aclrtCreateEvent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.aclrtCreateEvent.restype = ctypes.c_int
        lib.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
        lib.aclrtDestroyEvent.restype = ctypes.c_int
        lib.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.aclrtRecordEvent.restype = ctypes.c_int
        lib.aclrtSynchronizeEvent.argtypes = [ctypes.c_void_p]
        lib.aclrtSynchronizeEvent.restype = ctypes.c_int
        lib.aclrtEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.aclrtEventElapsedTime.restype = ctypes.c_int

    def measure_us(self, launch, repetitions: int) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.runtime._check(
            self.runtime.lib.aclrtCreateEvent(ctypes.byref(start)),
            "aclrtCreateEvent(start)",
        )
        self.runtime._check(
            self.runtime.lib.aclrtCreateEvent(ctypes.byref(end)),
            "aclrtCreateEvent(end)",
        )
        try:
            self.runtime._check(
                self.runtime.lib.aclrtRecordEvent(start, self.runtime.stream),
                "aclrtRecordEvent(start)",
            )
            for _ in range(repetitions):
                launch()
            self.runtime._check(
                self.runtime.lib.aclrtRecordEvent(end, self.runtime.stream),
                "aclrtRecordEvent(end)",
            )
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeEvent(end),
                "aclrtSynchronizeEvent",
            )
            elapsed_ms = ctypes.c_float()
            self.runtime._check(
                self.runtime.lib.aclrtEventElapsedTime(
                    ctypes.byref(elapsed_ms), start, end
                ),
                "aclrtEventElapsedTime",
            )
            return float(elapsed_ms.value) * 1000.0 / repetitions
        finally:
            self.runtime.lib.aclrtDestroyEvent(start)
            self.runtime.lib.aclrtDestroyEvent(end)


class Invocation:
    def __init__(self, runtime: AclRuntime, op: MhcSinkhornAclnn, spec: dict) -> None:
        self.runtime = runtime
        self.op = op
        self.spec = spec
        dtype = np.dtype(spec["dtype"])
        rng = np.random.default_rng(1701 + spec["matrix_count"] + spec["iterations"])
        logits_host = rng.normal(0.0, 2.0, size=spec["shape"]).astype(dtype)
        mask_host = None
        if spec["mask_mode"] == "scalar":
            mask_host = np.asarray([0.125], dtype=dtype)
        elif spec["mask_mode"] == "full":
            mask_host = rng.normal(0.0, 0.2, size=spec["shape"]).astype(dtype)
        elif spec["mask_mode"] != "absent":
            raise ValueError(f"unsupported mask mode: {spec['mask_mode']}")

        self.logits = DeviceArray.from_host(runtime, logits_host)
        self.mask = (
            DeviceArray.from_host(runtime, mask_host) if mask_host is not None else None
        )
        self.output = DeviceArray.empty(runtime, logits_host.shape, logits_host.dtype)
        self.logits_handle = op.tensor(self.logits)
        self.mask_handle = op.tensor(self.mask) if self.mask is not None else ctypes.c_void_p()
        self.output_handle = op.tensor(self.output)
        self.workspace = ctypes.c_void_p()
        self.workspace_size: int | None = None

    def launch(self) -> None:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        self.runtime._check(
            self.op.custom.aclnnMhcSinkhornGetWorkspaceSize(
                self.logits_handle,
                self.mask_handle,
                self.spec["iterations"],
                self.spec["eps"],
                self.output_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            ),
            "aclnnMhcSinkhornGetWorkspaceSize",
        )
        if not executor.value:
            raise RuntimeError("aclnnMhcSinkhornGetWorkspaceSize returned null executor")
        if self.workspace_size is None:
            self.workspace_size = workspace_size.value
            if workspace_size.value:
                self.workspace = self.runtime.malloc(workspace_size.value)
        elif self.workspace_size != workspace_size.value:
            raise RuntimeError("workspace size changed across identical launches")
        self.runtime._check(
            self.op.custom.aclnnMhcSinkhorn(
                self.workspace,
                workspace_size.value,
                executor,
                self.runtime.stream,
            ),
            "aclnnMhcSinkhorn",
        )

    def close(self) -> None:
        if self.workspace.value:
            self.runtime.lib.aclrtFree(self.workspace)
            self.workspace = ctypes.c_void_p()
        self.op.nnopbase.aclDestroyTensor(self.logits_handle)
        if self.mask_handle.value:
            self.op.nnopbase.aclDestroyTensor(self.mask_handle)
        self.op.nnopbase.aclDestroyTensor(self.output_handle)
        self.logits.close()
        if self.mask is not None:
            self.mask.close()
        self.output.close()


def repetitions_for(spec: dict) -> int:
    work = spec["matrix_count"] * spec["n"] * spec["n"] * spec["iterations"]
    if work <= 10_000:
        return 100
    if work <= 100_000:
        return 50
    if work <= 1_000_000:
        return 20
    return 10


def benchmark_case(
    runtime: AclRuntime,
    op: MhcSinkhornAclnn,
    timer: EventTimer,
    spec: dict,
) -> dict:
    invocation = Invocation(runtime, op, spec)
    try:
        for _ in range(WARMUP):
            invocation.launch()
        runtime._check(
            runtime.lib.aclrtSynchronizeStream(runtime.stream),
            "aclrtSynchronizeStream(warmup)",
        )
        repetitions = repetitions_for(spec)
        samples = [timer.measure_us(invocation.launch, repetitions) for _ in range(ACTIVE)]
    finally:
        invocation.close()

    samples.sort()
    median_us = statistics.median(samples)
    p90_index = max(0, math.ceil(0.9 * len(samples)) - 1)
    result = {
        **spec,
        "shape": list(spec["shape"]),
        "repetitions": repetitions,
        "samples_us": samples,
        "median_us": median_us,
        "min_us": samples[0],
        "p90_us": samples[p90_index],
        "latency_per_matrix_us": median_us / spec["matrix_count"],
        "matrices_per_us": spec["matrix_count"] / median_us,
    }
    print("PERF_CASE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def linear_fit(results: list[dict]) -> tuple[float, float, float] | None:
    points = sorted(
        (result["iterations"], result["median_us"])
        for result in results
        if "iterations" in result["tags"]
    )
    if len(points) < 2:
        return None
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total else 1.0
    return float(intercept), float(slope), r_squared


def comparison_lines(results: list[dict], tag: str, key: str) -> list[str]:
    selected = sorted(
        (result for result in results if tag in result["tags"]),
        key=lambda item: item[key],
    )
    return [
        f"- {key}={result[key]}: {result['median_us']:.4f} us "
        f"({result['latency_per_matrix_us']:.6f} us/matrix)"
        for result in selected
    ]


def write_report(path: Path, label: str, results: list[dict]) -> None:
    lines = [
        f"# MhcSinkhorn local A3 performance — {label}",
        "",
        "> Evidence class: `benchmark_observed`.",
        "> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.",
        f"> Warmup={WARMUP}; active samples={ACTIVE}; each active sample averages repeated identical ACLNN launches.",
        "",
        "| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        samples = ", ".join(f"{value:.4f}" for value in result["samples_us"])
        lines.append(
            f"| {result['id']} | {result['shape']} | {result['matrix_count']} | "
            f"{result['n']} | {result['iterations']} | {result['dtype']} | "
            f"{result['mask_mode']} | {result['median_us']:.4f} | "
            f"{result['latency_per_matrix_us']:.6f} | {result['matrices_per_us']:.4f} | {samples} |"
        )

    lines.extend(["", "## Matrix-count amortization", ""])
    lines.extend(comparison_lines(results, "matrix_count", "matrix_count"))
    lines.extend(["", "## Iteration cost model", ""])
    lines.extend(comparison_lines(results, "iterations", "iterations"))
    fit = linear_fit(results)
    if fit is not None:
        intercept, slope, r_squared = fit
        lines.extend(
            [
                "",
                f"Least-squares proxy: `T(iterations) = {intercept:.4f} us + iterations * {slope:.4f} us`, R²={r_squared:.6f}.",
            ]
        )
    lines.extend(["", "## Orthogonal comparisons", "", "### N", ""])
    lines.extend(comparison_lines(results, "n", "n"))
    lines.extend(["", "### DType", ""])
    lines.extend(comparison_lines(results, "dtype", "dtype"))
    lines.extend(["", "### Mask", ""])
    lines.extend(comparison_lines(results, "mask", "mask_mode"))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Tensor allocation, input copies, and output copies are outside the timed event interval.",
            "- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.",
            "- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested.difference(case["id"] for case in cases)
        if missing:
            parser.error(f"unknown performance case id(s): {', '.join(sorted(missing))}")

    runtime = AclRuntime()
    try:
        operator = MhcSinkhornAclnn(runtime)
        timer = EventTimer(runtime)
        results = [benchmark_case(runtime, operator, timer, spec) for spec in cases]
    finally:
        runtime.close()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, sort_keys=True) + "\n")
    write_report(args.output_report, args.label, results)
    total_median_us = sum(result["median_us"] for result in results)
    print(f"SCORE_TOTAL_MEDIAN_US={total_median_us:.6f}", flush=True)
    print(f"PERFORMANCE_JSONL={args.output_jsonl.resolve()}", flush=True)
    print(f"PERFORMANCE_REPORT={args.output_report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

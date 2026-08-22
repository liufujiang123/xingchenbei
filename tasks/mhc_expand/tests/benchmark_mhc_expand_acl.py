#!/usr/bin/env python3
"""Local A3 proxy benchmark using CANN ACL runtime events, not wall-clock time."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

from test_mhc_expand import AclRuntime, DeviceArray, MhcExpandAclnn, encode


WARMUP = 5
ACTIVE = 7


def load_perf_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if "perf" in case["tags"]:
            cases.append(case)
    return cases


def parse_case(case: dict) -> dict:
    inputs = {entry["name"]: entry for entry in case["inputs"]}
    shape = tuple(int(value) for value in inputs["x"]["shape"])
    dtype = {"float16": "fp16", "bfloat16": "bf16"}[inputs["x"]["dtype"]]
    backward = bool(inputs["backward"]["value"])
    m = int(inputs["mhc_mult"]["value"])
    s, d = shape[0], shape[-1]
    expected_shape = (s, d) if backward else (s, m, d)
    return {
        "id": case["id"],
        "shape": shape,
        "dtype": dtype,
        "backward": backward,
        "m": m,
        "s": s,
        "d": d,
        "output_shape": expected_shape,
    }


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
        self.runtime._check(self.runtime.lib.aclrtCreateEvent(ctypes.byref(start)), "aclrtCreateEvent(start)")
        self.runtime._check(self.runtime.lib.aclrtCreateEvent(ctypes.byref(end)), "aclrtCreateEvent(end)")
        try:
            self.runtime._check(self.runtime.lib.aclrtRecordEvent(start, self.runtime.stream), "aclrtRecordEvent(start)")
            for _ in range(repetitions):
                launch()
            self.runtime._check(self.runtime.lib.aclrtRecordEvent(end, self.runtime.stream), "aclrtRecordEvent(end)")
            self.runtime._check(self.runtime.lib.aclrtSynchronizeEvent(end), "aclrtSynchronizeEvent")
            elapsed_ms = ctypes.c_float()
            self.runtime._check(
                self.runtime.lib.aclrtEventElapsedTime(ctypes.byref(elapsed_ms), start, end),
                "aclrtEventElapsedTime",
            )
            return float(elapsed_ms.value) * 1000.0 / repetitions
        finally:
            self.runtime.lib.aclrtDestroyEvent(start)
            self.runtime.lib.aclrtDestroyEvent(end)


class Invocation:
    def __init__(self, runtime: AclRuntime, op: MhcExpandAclnn, spec: dict) -> None:
        self.runtime = runtime
        self.op = op
        self.spec = spec
        source = np.zeros(spec["shape"], dtype=np.float32)
        self.x_host = encode(source, spec["dtype"])
        output_dtype = np.uint16 if spec["dtype"] == "bf16" else np.float16
        self.o_host = np.empty(spec["output_shape"], dtype=output_dtype)
        self.x = DeviceArray.from_host(runtime, self.x_host)
        self.o = DeviceArray.empty_like(runtime, self.o_host)
        self.x_handle = op._tensor(spec["shape"], spec["dtype"], self.x.pointer)
        self.o_handle = op._tensor(spec["output_shape"], spec["dtype"], self.o.pointer)
        self.workspace = None
        self.workspace_size = None

    def launch(self) -> None:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        status = self.op.custom.aclnnMhcExpandGetWorkspaceSize(
            self.x_handle,
            self.spec["m"],
            self.spec["backward"],
            self.o_handle,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        self.runtime._check(status, "aclnnMhcExpandGetWorkspaceSize")
        if not executor.value:
            raise RuntimeError("aclnnMhcExpandGetWorkspaceSize returned null executor")
        if self.workspace_size is None:
            self.workspace_size = workspace_size.value
            if workspace_size.value:
                self.workspace = self.runtime.malloc(workspace_size.value)
        elif self.workspace_size != workspace_size.value:
            raise RuntimeError("workspace size changed across identical launches")
        workspace_pointer = self.workspace if self.workspace is not None else ctypes.c_void_p()
        self.runtime._check(
            self.op.custom.aclnnMhcExpand(
                workspace_pointer,
                workspace_size.value,
                executor,
                self.runtime.stream,
            ),
            "aclnnMhcExpand",
        )

    def close(self) -> None:
        if self.workspace is not None:
            self.runtime.free(self.workspace)
        self.op.nnopbase.aclDestroyTensor(self.x_handle)
        self.op.nnopbase.aclDestroyTensor(self.o_handle)
        self.x.close()
        self.o.close()


def repetitions_for(spec: dict) -> int:
    traffic_bytes = (spec["m"] + 1) * spec["s"] * spec["d"] * 2
    if traffic_bytes < 1 << 16:
        return 100
    if traffic_bytes < 1 << 22:
        return 30
    return 10


def benchmark_case(runtime: AclRuntime, op: MhcExpandAclnn, timer: EventTimer, spec: dict) -> dict:
    invocation = Invocation(runtime, op, spec)
    try:
        for _ in range(WARMUP):
            invocation.launch()
        runtime._check(runtime.lib.aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream(warmup)")
        repetitions = repetitions_for(spec)
        samples = [timer.measure_us(invocation.launch, repetitions) for _ in range(ACTIVE)]
    finally:
        invocation.close()

    samples.sort()
    median_us = statistics.median(samples)
    p90_index = max(0, math.ceil(0.9 * len(samples)) - 1)
    traffic_bytes = (spec["m"] + 1) * spec["s"] * spec["d"] * 2
    bandwidth_gbps = traffic_bytes / median_us / 1000.0
    result = {
        **spec,
        "mode": "backward" if spec["backward"] else "forward",
        "repetitions": repetitions,
        "median_us": median_us,
        "min_us": samples[0],
        "p90_us": samples[p90_index],
        "bandwidth_gbps": bandwidth_gbps,
    }
    print("PERF_CASE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def write_report(path: Path, label: str, results: list[dict]) -> None:
    lines = [
        f"# MhcExpand local A3 performance — {label}",
        "",
        "> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.",
        f"> Warmup={WARMUP}; active samples={ACTIVE}; each sample uses case-dependent repeated launches.",
        "",
        "| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['id']} | {result['mode']} | {list(result['shape'])} | {result['dtype']} | "
            f"{result['m']} | {result['median_us']:.4f} | {result['min_us']:.4f} | "
            f"{result['p90_us']:.4f} | {result['bandwidth_gbps']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Cases: {len(results)}",
            f"- Forward cases: {sum(not item['backward'] for item in results)}",
            f"- Backward cases: {sum(item['backward'] for item in results)}",
            "- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    cases = [parse_case(case) for case in load_perf_cases(args.cases)]
    runtime = AclRuntime()
    try:
        op = MhcExpandAclnn(runtime)
        timer = EventTimer(runtime)
        results = [benchmark_case(runtime, op, timer, spec) for spec in cases]
    finally:
        runtime.close()
    write_report(args.output, args.label, results)
    total_median_us = sum(result["median_us"] for result in results)
    print(f"SCORE_TOTAL_MEDIAN_US={total_median_us:.6f}", flush=True)
    print(f"PERFORMANCE_REPORT={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

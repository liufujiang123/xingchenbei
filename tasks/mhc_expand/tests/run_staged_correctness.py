#!/usr/bin/env python3
"""Staged, unbuffered MhcExpand correctness runner with API-level markers."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

import numpy as np

from test_mhc_expand import (
    ACL_FORMAT_ND,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_SUCCESS,
    DeviceArray,
    MhcExpandAclnn,
    bf16_bits_to_fp32,
    decode,
    dtype_code,
    encode,
)


def mark(name: str, **fields) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"{name} monotonic_s={time.monotonic():.9f}"
        f"{(' ' + suffix) if suffix else ''}",
        flush=True,
    )


class InstrumentedRuntime:
    def __init__(self) -> None:
        mark("RUNTIME_INIT_BEGIN")
        self.lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        self._bind()
        self.device = int(os.environ.get("MHC_EXPAND_DEVICE_ID", "0"))
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self.initialized = False
        self.device_set = False

        try:
            self._timed_check("ACL_INIT", self.lib.aclInit, None)
            self.initialized = True
            self._timed_check("ACLRT_SET_DEVICE", self.lib.aclrtSetDevice, self.device)
            self.device_set = True
            self._timed_check(
                "ACLRT_CREATE_CONTEXT",
                self.lib.aclrtCreateContext,
                ctypes.byref(self.context),
                self.device,
            )
            self._timed_check(
                "ACLRT_CREATE_STREAM",
                self.lib.aclrtCreateStream,
                ctypes.byref(self.stream),
            )
        except BaseException:
            self.close()
            raise
        mark("RUNTIME_INIT_END", device=self.device)

    def _bind(self) -> None:
        self.lib.aclInit.argtypes = [ctypes.c_char_p]
        self.lib.aclInit.restype = ctypes.c_int
        self.lib.aclFinalize.argtypes = []
        self.lib.aclFinalize.restype = ctypes.c_int
        self.lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.lib.aclrtSetDevice.restype = ctypes.c_int
        self.lib.aclrtResetDevice.argtypes = [ctypes.c_int32]
        self.lib.aclrtResetDevice.restype = ctypes.c_int
        self.lib.aclrtCreateContext.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self.lib.aclrtCreateContext.restype = ctypes.c_int
        self.lib.aclrtDestroyContext.argtypes = [ctypes.c_void_p]
        self.lib.aclrtDestroyContext.restype = ctypes.c_int
        self.lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.aclrtCreateStream.restype = ctypes.c_int
        self.lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        self.lib.aclrtDestroyStream.restype = ctypes.c_int
        self.lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        self.lib.aclrtSynchronizeStream.restype = ctypes.c_int
        self.lib.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.aclrtMalloc.restype = ctypes.c_int
        self.lib.aclrtFree.argtypes = [ctypes.c_void_p]
        self.lib.aclrtFree.restype = ctypes.c_int
        self.lib.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.aclrtMemcpy.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{operation} failed: {status}")

    def _timed_check(self, name: str, function, *args) -> None:
        mark(f"{name}_BEGIN")
        start = time.monotonic()
        status = int(function(*args))
        elapsed_ms = (time.monotonic() - start) * 1000.0
        mark(f"{name}_END", elapsed_ms=f"{elapsed_ms:.3f}", rc=status)
        self._check(status, name)

    def close(self) -> None:
        if self.stream.value:
            self._timed_check("ACLRT_DESTROY_STREAM", self.lib.aclrtDestroyStream, self.stream)
            self.stream = ctypes.c_void_p()
        if self.context.value:
            self._timed_check(
                "ACLRT_DESTROY_CONTEXT", self.lib.aclrtDestroyContext, self.context
            )
            self.context = ctypes.c_void_p()
        if self.device_set:
            self._timed_check("ACLRT_RESET_DEVICE", self.lib.aclrtResetDevice, self.device)
            self.device_set = False
        if self.initialized:
            self._timed_check("ACL_FINALIZE", self.lib.aclFinalize)
            self.initialized = False

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.aclrtMalloc(
                ctypes.byref(pointer), size, ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc",
        )
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self._check(self.lib.aclrtFree(pointer), "aclrtFree")

    def copy_to_device(self, destination: ctypes.c_void_p, source: np.ndarray) -> None:
        self._check(
            self.lib.aclrtMemcpy(
                destination,
                source.nbytes,
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "aclrtMemcpy(H2D)",
        )

    def copy_to_host(self, destination: np.ndarray, source: ctypes.c_void_p) -> None:
        self._check(
            self.lib.aclrtMemcpy(
                ctypes.c_void_p(destination.ctypes.data),
                destination.nbytes,
                source,
                destination.nbytes,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            "aclrtMemcpy(D2H)",
        )


class InstrumentedOp(MhcExpandAclnn):
    def __init__(self, runtime: InstrumentedRuntime) -> None:
        mark("CUSTOM_LIBRARY_LOAD_BEGIN")
        start = time.monotonic()
        super().__init__(runtime)
        mark(
            "CUSTOM_LIBRARY_LOAD_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            library=os.path.realpath(os.environ["MHC_EXPAND_CUSTOM_LIB"]),
        )


def deterministic_input(s: int, d: int, m: int, dtype: str, backward: bool):
    if backward and s == 1 and d == 16 and m == 2:
        source = np.empty((1, 2, 16), dtype=np.float32)
        source[:, 0, :] = 1.0
        source[:, 1, :] = 2.0
    elif backward:
        values = np.arange(s * m * d, dtype=np.float32).reshape(s, m, d)
        source = ((values % 7.0) - 3.0) / 4.0
    else:
        values = np.arange(s * d, dtype=np.float32).reshape(s, d)
        source = ((values % 31.0) - 15.0) if d != 16 else values
    return encode(source, dtype)


def reference(x: np.ndarray, dtype: str, m: int, backward: bool) -> np.ndarray:
    if backward:
        return encode(decode(x, dtype).sum(axis=1, dtype=np.float32), dtype)
    return np.repeat(x[:, None, :], m, axis=1)


def output_shape(s: int, d: int, m: int, backward: bool) -> tuple[int, ...]:
    return (s, d) if backward else (s, m, d)


def run_case(
    runtime: InstrumentedRuntime,
    op: InstrumentedOp,
    *,
    name: str,
    s: int,
    d: int,
    m: int,
    dtype: str,
    backward: bool,
) -> dict:
    print(
        f"CASE_BEGIN name={name} dtype={dtype} S={s} D={d} m={m} "
        f"backward={str(backward).lower()}",
        flush=True,
    )
    x_host = deterministic_input(s, d, m, dtype, backward)
    expected = reference(x_host, dtype, m, backward)
    out_host = np.empty(output_shape(s, d, m, backward), dtype=x_host.dtype)

    mark("INPUT_ALLOCATION_BEGIN")
    x = DeviceArray.from_host(runtime, x_host)
    out = DeviceArray.empty_like(runtime, out_host)
    mark("INPUT_ALLOCATION_END", input_bytes=x_host.nbytes, output_bytes=out_host.nbytes)

    mark("ACL_CREATE_TENSOR_BEGIN")
    x_handle = op._tensor(tuple(x_host.shape), dtype, x.pointer)
    out_handle = op._tensor(tuple(out_host.shape), dtype, out.pointer)
    mark("ACL_CREATE_TENSOR_END")
    workspace = None
    try:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        mark("GET_WORKSPACE_BEGIN")
        start = time.monotonic()
        host_rc = op.custom.aclnnMhcExpandGetWorkspaceSize(
            x_handle,
            m,
            backward,
            out_handle,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        mark(
            "GET_WORKSPACE_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=host_rc,
            workspace_size=workspace_size.value,
            executor_nonnull=bool(executor.value),
        )
        runtime._check(host_rc, "aclnnMhcExpandGetWorkspaceSize")

        workspace_pointer = ctypes.c_void_p()
        if workspace_size.value:
            workspace = runtime.malloc(workspace_size.value)
            workspace_pointer = workspace

        mark("KERNEL_LAUNCH_BEGIN")
        start = time.monotonic()
        launch_rc = op.custom.aclnnMhcExpand(
            workspace_pointer,
            workspace_size.value,
            executor,
            runtime.stream,
        )
        mark(
            "KERNEL_LAUNCH_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=launch_rc,
        )
        runtime._check(launch_rc, "aclnnMhcExpand")

        mark("SYNCHRONIZE_BEGIN")
        start = time.monotonic()
        sync_rc = runtime.lib.aclrtSynchronizeStream(runtime.stream)
        mark(
            "SYNCHRONIZE_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=sync_rc,
        )
        runtime._check(sync_rc, "aclrtSynchronizeStream")

        mark("COPY_BACK_BEGIN")
        start = time.monotonic()
        actual = out.fetch().copy()
        mark(
            "COPY_BACK_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
        )
    finally:
        if workspace is not None:
            runtime.free(workspace)
        op.nnopbase.aclDestroyTensor(x_handle)
        op.nnopbase.aclDestroyTensor(out_handle)
        x.close()
        out.close()

    actual_float = decode(actual, dtype)
    expected_float = decode(expected, dtype)
    abs_error = np.abs(actual_float - expected_float)
    denominator = np.maximum(np.abs(expected_float), np.float32(1.0e-12))
    rel_error = abs_error / denominator
    result = {
        "name": name,
        "dtype": dtype,
        "s": s,
        "d": d,
        "m": m,
        "backward": backward,
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "max_rel_error": float(rel_error.max(initial=0.0)),
        "exact_equal": bool(np.array_equal(actual, expected)),
    }
    if actual.size <= 64:
        print(f"input={x_host.tolist()}", flush=True)
        print(f"actual={actual.tolist()}", flush=True)
        print(f"expected={expected.tolist()}", flush=True)
    print(f"CASE_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    if not result["exact_equal"]:
        np.set_printoptions(threshold=np.inf)
        print(f"FAIL_INPUT={x_host}", flush=True)
        print(f"FAIL_ACTUAL={actual}", flush=True)
        print(f"FAIL_EXPECTED={expected}", flush=True)
        raise AssertionError(f"{name} did not match reference exactly")
    return result


PHASES = {
    "forward_fp16_smoke": [
        ("forward_fp16_smoke", 1, 16, 2, "fp16", False),
    ],
    "forward_bf16_smoke": [
        ("forward_bf16_smoke", 1, 16, 2, "bf16", False),
    ],
    "forward_boundary": [
        ("forward_fp16_d1", 1, 1, 2, "fp16", False),
        ("forward_fp16_d17", 3, 17, 2, "fp16", False),
        ("forward_bf16_d33", 3, 33, 4, "bf16", False),
        ("forward_bf16_d129", 5, 129, 4, "bf16", False),
    ],
    "backward_fp16_smoke": [
        ("backward_fp16_smoke", 1, 16, 2, "fp16", True),
    ],
    "backward_bf16_smoke": [
        ("backward_bf16_smoke", 1, 16, 2, "bf16", True),
    ],
    "backward_boundary": [
        ("backward_fp16_d1", 1, 1, 2, "fp16", True),
        ("backward_fp16_d17", 3, 17, 2, "fp16", True),
        ("backward_bf16_d33", 3, 33, 4, "bf16", True),
        ("backward_bf16_d129", 5, 129, 8, "bf16", True),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    args = parser.parse_args()

    runtime = InstrumentedRuntime()
    try:
        op = InstrumentedOp(runtime)
        results = []
        for name, s, d, m, dtype, backward in PHASES[args.phase]:
            results.append(
                run_case(
                    runtime,
                    op,
                    name=name,
                    s=s,
                    d=d,
                    m=m,
                    dtype=dtype,
                    backward=backward,
                )
            )
        print(f"PHASE_PASS {json.dumps(results, sort_keys=True)}", flush=True)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run an isolated built-in aclnnAbs control without loading any custom OPP."""

from __future__ import annotations

import ctypes
import json
import os
import time

import numpy as np

ACL_FORMAT_ND = 2
ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def mark(name: str, **fields) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"{name}{(' ' + suffix) if suffix else ''}", flush=True)


class InstrumentedRuntime:
    def __init__(self) -> None:
        mark("RUNTIME_INIT_BEGIN")
        self.lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        self.device = int(os.environ.get("MHC_EXPAND_DEVICE_ID", "0"))
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self.initialized = False
        self.device_set = False
        self._bind()
        self._timed_check("ACL_INIT", self.lib.aclInit, None)
        self.initialized = True
        try:
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
    def check(rc: int, operation: str) -> None:
        if rc != ACL_SUCCESS:
            raise RuntimeError(f"{operation} failed: {rc}")

    def _timed_check(self, name: str, function, *args) -> None:
        mark(f"{name}_BEGIN")
        start = time.monotonic()
        rc = int(function(*args))
        mark(f"{name}_END", elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}", rc=rc)
        self.check(rc, name)

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.aclrtMalloc(
                ctypes.byref(pointer), size, ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc",
        )
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self.check(self.lib.aclrtFree(pointer), "aclrtFree")

    def close(self) -> None:
        if self.stream.value:
            self._timed_check("ACLRT_DESTROY_STREAM", self.lib.aclrtDestroyStream, self.stream)
            self.stream = ctypes.c_void_p()
        if self.context.value:
            self._timed_check("ACLRT_DESTROY_CONTEXT", self.lib.aclrtDestroyContext, self.context)
            self.context = ctypes.c_void_p()
        if self.device_set:
            self._timed_check("ACLRT_RESET_DEVICE", self.lib.aclrtResetDevice, self.device)
            self.device_set = False
        if self.initialized:
            self._timed_check("ACL_FINALIZE", self.lib.aclFinalize)
            self.initialized = False


class DeviceArray:
    def __init__(self, runtime: InstrumentedRuntime, host: np.ndarray, copy: bool) -> None:
        self.runtime = runtime
        self.host = np.ascontiguousarray(host)
        self.pointer = runtime.malloc(self.host.nbytes)
        if copy:
            runtime.check(
                runtime.lib.aclrtMemcpy(
                    self.pointer,
                    self.host.nbytes,
                    ctypes.c_void_p(self.host.ctypes.data),
                    self.host.nbytes,
                    ACL_MEMCPY_HOST_TO_DEVICE,
                ),
                "aclrtMemcpy(H2D)",
            )

    @classmethod
    def from_host(cls, runtime: InstrumentedRuntime, host: np.ndarray) -> "DeviceArray":
        return cls(runtime, host, True)

    @classmethod
    def empty_like(cls, runtime: InstrumentedRuntime, host: np.ndarray) -> "DeviceArray":
        return cls(runtime, np.empty_like(host), False)

    def fetch(self) -> np.ndarray:
        self.runtime.check(
            self.runtime.lib.aclrtMemcpy(
                ctypes.c_void_p(self.host.ctypes.data),
                self.host.nbytes,
                self.pointer,
                self.host.nbytes,
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
            "aclrtMemcpy(D2H)",
        )
        return self.host

    def close(self) -> None:
        if self.pointer.value:
            self.runtime.free(self.pointer)
            self.pointer = ctypes.c_void_p()


def create_tensor(
    nnop: ctypes.CDLL,
    shape: tuple[int, ...],
    pointer: ctypes.c_void_p,
) -> ctypes.c_void_p:
    dims = (ctypes.c_int64 * len(shape))(*shape)
    strides_list: list[int] = []
    stride = 1
    for dimension in reversed(shape):
        strides_list.append(stride)
        stride *= dimension
    strides = (ctypes.c_int64 * len(shape))(*reversed(strides_list))
    handle = nnop.aclCreateTensor(
        dims,
        len(shape),
        1,  # ACL_FLOAT16
        strides,
        0,
        ACL_FORMAT_ND,
        dims,
        len(shape),
        pointer,
    )
    if not handle:
        raise RuntimeError("aclCreateTensor returned null")
    return ctypes.c_void_p(handle)


def bind(nnop: ctypes.CDLL, opapi: ctypes.CDLL) -> None:
    nnop.aclCreateTensor.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    nnop.aclCreateTensor.restype = ctypes.c_void_p
    nnop.aclDestroyTensor.argtypes = [ctypes.c_void_p]
    nnop.aclDestroyTensor.restype = ctypes.c_int
    opapi.aclnnAbsGetWorkspaceSize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    opapi.aclnnAbsGetWorkspaceSize.restype = ctypes.c_int
    opapi.aclnnAbs.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    opapi.aclnnAbs.restype = ctypes.c_int


def main() -> int:
    custom_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    custom_lib = os.environ.get("MHC_EXPAND_CUSTOM_LIB", "")
    print(
        "CONTROL_ENV "
        + json.dumps(
            {
                "pid": os.getpid(),
                "device": os.environ.get("MHC_EXPAND_DEVICE_ID", "0"),
                "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH", ""),
                "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH", ""),
                "ASCEND_CUSTOM_OPP_PATH": custom_opp,
                "MHC_EXPAND_CUSTOM_LIB": custom_lib,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if custom_opp or custom_lib:
        raise RuntimeError("built-in control requires an empty custom OPP/library environment")

    runtime = InstrumentedRuntime()
    x: DeviceArray | None = None
    out: DeviceArray | None = None
    workspace: ctypes.c_void_p | None = None
    x_handle = ctypes.c_void_p()
    out_handle = ctypes.c_void_p()
    nnop = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    opapi = ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)
    bind(nnop, opapi)
    try:
        source = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float16)
        expected = np.abs(source)
        x = DeviceArray.from_host(runtime, source)
        out = DeviceArray.empty_like(runtime, np.empty_like(source))
        x_handle = create_tensor(nnop, source.shape, x.pointer)
        out_handle = create_tensor(nnop, source.shape, out.pointer)

        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        mark("BUILTIN_ABS_GET_WORKSPACE_BEGIN")
        start = time.monotonic()
        rc = int(
            opapi.aclnnAbsGetWorkspaceSize(
                x_handle,
                out_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        )
        mark(
            "BUILTIN_ABS_GET_WORKSPACE_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=rc,
            workspace_size=workspace_size.value,
            executor_nonnull=bool(executor.value),
        )
        if rc != ACL_SUCCESS or not executor.value:
            return 1

        workspace_pointer = ctypes.c_void_p()
        if workspace_size.value:
            workspace = runtime.malloc(workspace_size.value)
            workspace_pointer = workspace

        mark("BUILTIN_ABS_LAUNCH_BEGIN")
        start = time.monotonic()
        rc = int(
            opapi.aclnnAbs(
                workspace_pointer,
                workspace_size.value,
                executor,
                runtime.stream,
            )
        )
        mark(
            "BUILTIN_ABS_LAUNCH_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=rc,
        )
        if rc != ACL_SUCCESS:
            return 1

        mark("BUILTIN_ABS_SYNCHRONIZE_BEGIN")
        start = time.monotonic()
        rc = int(runtime.lib.aclrtSynchronizeStream(runtime.stream))
        mark(
            "BUILTIN_ABS_SYNCHRONIZE_END",
            elapsed_ms=f"{(time.monotonic() - start) * 1000.0:.3f}",
            rc=rc,
        )
        if rc != ACL_SUCCESS:
            return 1

        mark("BUILTIN_ABS_COPY_BACK_BEGIN")
        actual = out.fetch().copy()
        mark("BUILTIN_ABS_COPY_BACK_END")
        exact_equal = bool(np.array_equal(actual, expected))
        print(f"BUILTIN_ABS_INPUT {source.tolist()}", flush=True)
        print(f"BUILTIN_ABS_EXPECTED {expected.tolist()}", flush=True)
        print(f"BUILTIN_ABS_ACTUAL {actual.tolist()}", flush=True)
        print(f"BUILTIN_ABS_EXACT_EQUAL {str(exact_equal).lower()}", flush=True)
        if not exact_equal:
            return 1
        print("BUILTIN_ACLNN_CONTROL_PASS", flush=True)
        return 0
    finally:
        if workspace is not None:
            runtime.free(workspace)
        if x_handle.value:
            nnop.aclDestroyTensor(x_handle)
        if out_handle.value:
            nnop.aclDestroyTensor(out_handle)
        if x is not None:
            x.close()
        if out is not None:
            out.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

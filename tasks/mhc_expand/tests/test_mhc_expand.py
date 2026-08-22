"""Direct ACLNN correctness tests for the MhcExpand competition operator."""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ACL_SUCCESS = 0
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_BF16 = 27
ACL_FORMAT_ND = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0


def fp32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_fp32(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.uint16).astype(np.uint32)
    return (bits << np.uint32(16)).view(np.float32)


def dtype_code(dtype: str) -> int:
    if dtype == "fp16":
        return ACL_FLOAT16
    if dtype == "bf16":
        return ACL_BF16
    if dtype == "fp32":
        return ACL_FLOAT
    raise ValueError(f"unsupported test dtype: {dtype}")


def encode(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp16":
        return np.ascontiguousarray(value, dtype=np.float16)
    if dtype == "bf16":
        return np.ascontiguousarray(fp32_to_bf16_bits(value))
    if dtype == "fp32":
        return np.ascontiguousarray(value, dtype=np.float32)
    raise ValueError(dtype)


def decode(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return bf16_bits_to_fp32(value)
    return value.astype(np.float32)


class AclRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        self.lib.aclInit.argtypes = [ctypes.c_char_p]
        self.lib.aclInit.restype = ctypes.c_int
        self.lib.aclFinalize.argtypes = []
        self.lib.aclFinalize.restype = ctypes.c_int
        self.lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.lib.aclrtSetDevice.restype = ctypes.c_int
        self.lib.aclrtResetDevice.argtypes = [ctypes.c_int32]
        self.lib.aclrtResetDevice.restype = ctypes.c_int
        self.lib.aclrtCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
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
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int
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

        self.device = int(os.environ.get("MHC_EXPAND_DEVICE_ID", "0"))
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self._check(self.lib.aclInit(None), "aclInit")
        try:
            self._check(self.lib.aclrtSetDevice(self.device), "aclrtSetDevice")
            self._check(
                self.lib.aclrtCreateContext(ctypes.byref(self.context), self.device),
                "aclrtCreateContext",
            )
            self._check(
                self.lib.aclrtCreateStream(ctypes.byref(self.stream)),
                "aclrtCreateStream",
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{operation} failed: {status}")

    def close(self) -> None:
        if self.stream.value:
            self._check(self.lib.aclrtDestroyStream(self.stream), "aclrtDestroyStream")
            self.stream = ctypes.c_void_p()
        if self.context.value:
            self._check(self.lib.aclrtDestroyContext(self.context), "aclrtDestroyContext")
            self.context = ctypes.c_void_p()
        self.lib.aclrtResetDevice(self.device)
        self.lib.aclFinalize()

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


@dataclass
class DeviceArray:
    runtime: AclRuntime
    host: np.ndarray
    pointer: ctypes.c_void_p

    @classmethod
    def from_host(cls, runtime: AclRuntime, host: np.ndarray) -> "DeviceArray":
        host = np.ascontiguousarray(host)
        pointer = runtime.malloc(host.nbytes)
        runtime.copy_to_device(pointer, host)
        return cls(runtime, host, pointer)

    @classmethod
    def empty_like(cls, runtime: AclRuntime, host: np.ndarray) -> "DeviceArray":
        host = np.empty_like(host)
        return cls(runtime, host, runtime.malloc(host.nbytes))

    def fetch(self) -> np.ndarray:
        self.runtime.copy_to_host(self.host, self.pointer)
        return self.host

    def close(self) -> None:
        if self.pointer.value:
            self.runtime.free(self.pointer)
            self.pointer = ctypes.c_void_p()


class MhcExpandAclnn:
    def __init__(self, runtime: AclRuntime) -> None:
        library_path = os.environ.get("MHC_EXPAND_CUSTOM_LIB")
        if not library_path or not os.path.isfile(library_path):
            raise RuntimeError("MHC_EXPAND_CUSTOM_LIB must point to libcust_opapi.so")
        self.runtime = runtime
        self.nnopbase = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
        self.custom = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        self.nnopbase.aclCreateTensor.argtypes = [
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
        self.nnopbase.aclCreateTensor.restype = ctypes.c_void_p
        self.nnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyTensor.restype = ctypes.c_int
        self.custom.aclnnMhcExpandGetWorkspaceSize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.custom.aclnnMhcExpandGetWorkspaceSize.restype = ctypes.c_int
        self.custom.aclnnMhcExpand.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.custom.aclnnMhcExpand.restype = ctypes.c_int

    def _tensor(
        self, shape: tuple[int, ...], dtype: str, pointer: ctypes.c_void_p
    ) -> ctypes.c_void_p:
        dims = (ctypes.c_int64 * len(shape))(*shape)
        stride_values = []
        stride = 1
        for dimension in reversed(shape):
            stride_values.append(stride)
            stride *= dimension
        strides = (ctypes.c_int64 * len(shape))(*reversed(stride_values))
        handle = self.nnopbase.aclCreateTensor(
            dims,
            len(shape),
            dtype_code(dtype),
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

    def workspace_status(
        self,
        x: DeviceArray,
        x_shape: tuple[int, ...],
        out: DeviceArray,
        out_shape: tuple[int, ...],
        dtype: str,
        mhc_mult: int,
        backward: bool,
    ) -> int:
        x_handle = self._tensor(x_shape, dtype, x.pointer)
        out_handle = self._tensor(out_shape, dtype, out.pointer)
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        try:
            return self.custom.aclnnMhcExpandGetWorkspaceSize(
                x_handle,
                mhc_mult,
                backward,
                out_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
        finally:
            self.nnopbase.aclDestroyTensor(x_handle)
            self.nnopbase.aclDestroyTensor(out_handle)

    def run(
        self,
        x_host: np.ndarray,
        output_shape: tuple[int, ...],
        dtype: str,
        mhc_mult: int,
        backward: bool,
    ) -> np.ndarray:
        output_host = np.empty(output_shape, dtype=x_host.dtype)
        x = DeviceArray.from_host(self.runtime, x_host)
        out = DeviceArray.empty_like(self.runtime, output_host)
        workspace = None
        x_handle = self._tensor(tuple(x_host.shape), dtype, x.pointer)
        out_handle = self._tensor(output_shape, dtype, out.pointer)
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        try:
            status = self.custom.aclnnMhcExpandGetWorkspaceSize(
                x_handle,
                mhc_mult,
                backward,
                out_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"aclnnMhcExpandGetWorkspaceSize failed: {status}")
            workspace_pointer = ctypes.c_void_p()
            if workspace_size.value:
                workspace = self.runtime.malloc(workspace_size.value)
                workspace_pointer = workspace
            status = self.custom.aclnnMhcExpand(
                workspace_pointer,
                workspace_size.value,
                executor,
                self.runtime.stream,
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"aclnnMhcExpand failed: {status}")
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeStream(self.runtime.stream),
                "aclrtSynchronizeStream",
            )
            return out.fetch().copy()
        finally:
            if workspace is not None:
                self.runtime.free(workspace)
            self.nnopbase.aclDestroyTensor(x_handle)
            self.nnopbase.aclDestroyTensor(out_handle)
            x.close()
            out.close()


@pytest.fixture(scope="session")
def runtime() -> AclRuntime:
    session = AclRuntime()
    yield session
    session.close()


@pytest.fixture(scope="session")
def op(runtime: AclRuntime) -> MhcExpandAclnn:
    return MhcExpandAclnn(runtime)


def load_matrix_cases() -> list[tuple[str, int, int, int, str, bool]]:
    cases = []
    path = Path(__file__).with_name("mhc_expand_perf_cases.jsonl")
    dtype_names = {"float16": "fp16", "bfloat16": "bf16"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if "correctness" not in case["tags"]:
            continue
        inputs = {entry["name"]: entry for entry in case["inputs"]}
        shape = inputs["x"]["shape"]
        backward = bool(inputs["backward"]["value"])
        m = int(inputs["mhc_mult"]["value"])
        cases.append(
            (
                case["id"],
                int(shape[0]),
                int(shape[-1]),
                m,
                dtype_names[inputs["x"]["dtype"]],
                backward,
            )
        )
    return cases


MATRIX_CASES = load_matrix_cases()
FORWARD_CASES = [case[:-1] for case in MATRIX_CASES if not case[-1]]


@pytest.mark.parametrize("case_id,s,d,m,dtype", FORWARD_CASES, ids=lambda value: str(value))
def test_forward(
    op: MhcExpandAclnn, case_id: str, s: int, d: int, m: int, dtype: str
) -> None:
    source = np.random.default_rng(1).standard_normal((s, d), dtype=np.float32)
    x = encode(source, dtype)
    actual = op.run(x, (s, m, d), dtype, m, False)
    expected = np.repeat(x[:, None, :], m, axis=1)
    assert np.array_equal(actual, expected)


BACKWARD_CASES = [case[:-1] for case in MATRIX_CASES if case[-1]]


@pytest.mark.parametrize("case_id,s,d,m,dtype", BACKWARD_CASES, ids=lambda value: str(value))
def test_backward(
    op: MhcExpandAclnn, case_id: str, s: int, d: int, m: int, dtype: str
) -> None:
    source = np.random.default_rng(2).standard_normal((s, m, d), dtype=np.float32)
    x = encode(source, dtype)
    actual = op.run(x, (s, d), dtype, m, True)
    expected = encode(decode(x, dtype).sum(axis=1, dtype=np.float32), dtype)
    assert np.array_equal(actual, expected), (
        f"max_abs_error={np.max(np.abs(decode(actual, dtype) - decode(expected, dtype)))}"
    )


def test_generated_api_uses_declared_mhc_mult_default_value(
    op: MhcExpandAclnn,
) -> None:
    source = np.random.default_rng(3).standard_normal((3, 17), dtype=np.float32)
    x = encode(source, "fp16")
    actual = op.run(x, (3, 2, 17), "fp16", 2, False)
    assert np.array_equal(actual, np.repeat(x[:, None, :], 2, axis=1))


@pytest.mark.parametrize(
    "shape,out_shape,m,backward",
    [
        ((2, 3), (2, 2, 3), 2, True),
        ((2, 2, 3), (2, 3), 2, False),
        ((2, 3, 4), (2, 4), 2, True),
        ((2, 3, 4, 5), (2, 5), 3, True),
    ],
)
def test_invalid_contract_is_rejected(
    op: MhcExpandAclnn,
    runtime: AclRuntime,
    shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    m: int,
    backward: bool,
) -> None:
    x = DeviceArray.from_host(runtime, np.ones(shape, dtype=np.float16))
    out = DeviceArray.empty_like(runtime, np.empty(out_shape, dtype=np.float16))
    try:
        assert op.workspace_status(
            x, shape, out, out_shape, "fp16", m, backward
        ) != ACL_SUCCESS
    finally:
        x.close()
        out.close()


def test_invalid_dtype_is_rejected(
    op: MhcExpandAclnn, runtime: AclRuntime
) -> None:
    x = DeviceArray.from_host(runtime, np.ones((2, 3), dtype=np.float32))
    out = DeviceArray.empty_like(runtime, np.empty((2, 2, 3), dtype=np.float32))
    try:
        assert op.workspace_status(
            x, (2, 3), out, (2, 2, 3), "fp32", 2, False
        ) != ACL_SUCCESS
    finally:
        x.close()
        out.close()

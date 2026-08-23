"""Direct ACLNN-v2 runtime comparison against the MhcSinkhorn CPU reference."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np
import pytest

from reference_mhc_sinkhorn import mhc_sinkhorn_reference


ACL_SUCCESS = 0
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_FORMAT_ND = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0


def dtype_code(dtype: np.dtype) -> int:
    return ACL_FLOAT16 if dtype == np.dtype(np.float16) else ACL_FLOAT


class AclRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        self.device = int(os.environ.get("MHC_SINKHORN_DEVICE_ID", "4"))
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self._bind()
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

    def _bind(self) -> None:
        self.lib.aclInit.argtypes = [ctypes.c_char_p]
        self.lib.aclInit.restype = ctypes.c_int
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

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{operation} failed: {status}")

    def close(self) -> None:
        if self.stream.value:
            self.lib.aclrtDestroyStream(self.stream)
            self.stream = ctypes.c_void_p()
        if self.context.value:
            self.lib.aclrtDestroyContext(self.context)
            self.context = ctypes.c_void_p()
        self.lib.aclrtResetDevice(self.device)
        self.lib.aclFinalize()

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.aclrtMalloc(ctypes.byref(pointer), size, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc",
        )
        return pointer

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
    def empty(cls, runtime: AclRuntime, shape, dtype) -> "DeviceArray":
        host = np.empty(shape, dtype=dtype)
        return cls(runtime, host, runtime.malloc(host.nbytes))

    def close(self) -> None:
        if self.pointer.value:
            self.runtime.lib.aclrtFree(self.pointer)
            self.pointer = ctypes.c_void_p()


class MhcSinkhornAclnn:
    def __init__(self, runtime: AclRuntime) -> None:
        library = os.environ.get("MHC_SINKHORN_CUSTOM_LIB")
        if not library or not os.path.isfile(library):
            raise RuntimeError("MHC_SINKHORN_CUSTOM_LIB must point to libcust_opapi.so")
        self.runtime = runtime
        self.nnopbase = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
        self.custom = ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
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
        self.custom.aclnnMhcSinkhornGetWorkspaceSize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.custom.aclnnMhcSinkhornGetWorkspaceSize.restype = ctypes.c_int
        self.custom.aclnnMhcSinkhorn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.custom.aclnnMhcSinkhorn.restype = ctypes.c_int

    def tensor(self, array: DeviceArray) -> ctypes.c_void_p:
        shape = tuple(array.host.shape)
        dims = (ctypes.c_int64 * len(shape))(*shape)
        stride = 1
        reversed_strides = []
        for dimension in reversed(shape):
            reversed_strides.append(stride)
            stride *= dimension
        strides = (ctypes.c_int64 * len(shape))(*reversed(reversed_strides))
        handle = self.nnopbase.aclCreateTensor(
            dims,
            len(shape),
            dtype_code(array.host.dtype),
            strides,
            0,
            ACL_FORMAT_ND,
            dims,
            len(shape),
            array.pointer,
        )
        if not handle:
            raise RuntimeError("aclCreateTensor returned null")
        return ctypes.c_void_p(handle)

    def run(
        self,
        logits_host: np.ndarray,
        mask_host: np.ndarray | None,
        iterations: int,
        eps: float,
    ) -> np.ndarray:
        logits = DeviceArray.from_host(self.runtime, logits_host)
        mask = DeviceArray.from_host(self.runtime, mask_host) if mask_host is not None else None
        output = DeviceArray.empty(self.runtime, logits_host.shape, logits_host.dtype)
        logits_handle = self.tensor(logits)
        mask_handle = self.tensor(mask) if mask is not None else ctypes.c_void_p()
        output_handle = self.tensor(output)
        workspace = ctypes.c_void_p()
        try:
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            self.runtime._check(
                self.custom.aclnnMhcSinkhornGetWorkspaceSize(
                    logits_handle,
                    mask_handle,
                    iterations,
                    eps,
                    output_handle,
                    ctypes.byref(workspace_size),
                    ctypes.byref(executor),
                ),
                "aclnnMhcSinkhornGetWorkspaceSize",
            )
            if workspace_size.value:
                workspace = self.runtime.malloc(workspace_size.value)
            self.runtime._check(
                self.custom.aclnnMhcSinkhorn(
                    workspace, workspace_size.value, executor, self.runtime.stream
                ),
                "aclnnMhcSinkhorn",
            )
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeStream(self.runtime.stream),
                "aclrtSynchronizeStream",
            )
            self.runtime.copy_to_host(output.host, output.pointer)
            return output.host.copy()
        finally:
            if workspace.value:
                self.runtime.lib.aclrtFree(workspace)
            self.nnopbase.aclDestroyTensor(logits_handle)
            if mask_handle.value:
                self.nnopbase.aclDestroyTensor(mask_handle)
            self.nnopbase.aclDestroyTensor(output_handle)
            logits.close()
            if mask is not None:
                mask.close()
            output.close()


@pytest.fixture(scope="module")
def op():
    runtime = AclRuntime()
    operator = MhcSinkhornAclnn(runtime)
    try:
        yield operator
    finally:
        runtime.close()


CASES = []
for dtype in (np.float16, np.float32):
    for n in (4, 6, 8):
        for mask_mode in (0, 1, 2):
            CASES.append((dtype, n, mask_mode))


@pytest.mark.parametrize("dtype,n,mask_mode", CASES)
def test_contract_matrix(op, dtype, n, mask_mode):
    prefix = () if n == 4 else ((2,) if n == 6 else (2, 2))
    shape = (*prefix, n, n)
    rng = np.random.default_rng(9000 + n * 10 + mask_mode)
    logits = rng.normal(0.0, 2.5, size=shape).astype(dtype)
    mask = None
    if mask_mode == 1:
        mask = np.asarray([0.125], dtype=dtype)
    elif mask_mode == 2:
        mask = rng.normal(0.0, 0.25, size=(logits.size,)).astype(dtype)
    expected = mhc_sinkhorn_reference(logits, mask, iterations=20, eps=1.0e-6)
    actual = op.run(logits, mask, 20, 1.0e-6)
    atol = rtol = 5.0e-3 if dtype is np.float16 else 6.0e-5
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype,shape,iterations,eps",
    [
        (np.float16, (3, 4, 4), 1, 1.0e-6),
        (np.float32, (2, 3, 6, 6), 3, 1.0e-4),
        (np.float32, (8, 8), 7, 1.0e-8),
    ],
)
def test_nondefault_and_extreme(op, dtype, shape, iterations, eps):
    values = np.linspace(-80.0, 80.0, np.prod(shape), dtype=np.float32).reshape(shape)
    logits = values.astype(dtype)
    expected = mhc_sinkhorn_reference(logits, iterations=iterations, eps=eps)
    actual = op.run(logits, None, iterations, eps)
    atol = rtol = 6.0e-3 if dtype is np.float16 else 8.0e-5
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

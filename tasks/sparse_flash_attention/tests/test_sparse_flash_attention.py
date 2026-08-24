#!/usr/bin/env python3
"""Deterministic direct-ACLNN correctness runner for SparseFlashAttention."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np


ACL_SUCCESS = 0
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3
ACL_BF16 = 27
ACL_FORMAT_ND = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
EMPTY_MAX = -np.finfo(np.float32).max


def dtype_code(array: np.ndarray) -> int:
    if array.dtype == np.float16:
        return ACL_FLOAT16
    if array.dtype == np.float32:
        return ACL_FLOAT
    if array.dtype == np.int32:
        return ACL_INT32
    raise ValueError(f"unsupported dtype: {array.dtype}")


class AclRuntime:
    def __init__(self, device: int) -> None:
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
        self.lib.aclrtCreateEvent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.aclrtCreateEvent.restype = ctypes.c_int
        self.lib.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
        self.lib.aclrtDestroyEvent.restype = ctypes.c_int
        self.lib.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.aclrtRecordEvent.restype = ctypes.c_int
        self.lib.aclrtSynchronizeEvent.argtypes = [ctypes.c_void_p]
        self.lib.aclrtSynchronizeEvent.restype = ctypes.c_int
        self.lib.aclrtEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.aclrtEventElapsedTime.restype = ctypes.c_int
        self.lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
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

        self.device = device
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self._check(self.lib.aclInit(None), "aclInit")
        try:
            self._check(self.lib.aclrtSetDevice(device), "aclrtSetDevice")
            self._check(
                self.lib.aclrtCreateContext(ctypes.byref(self.context), device),
                "aclrtCreateContext",
            )
            self._check(self.lib.aclrtCreateStream(ctypes.byref(self.stream)), "aclrtCreateStream")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != ACL_SUCCESS:
            raise RuntimeError(f"{operation} failed: {status}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(
            self.lib.aclrtMalloc(ctypes.byref(pointer), size, ACL_MEM_MALLOC_HUGE_FIRST),
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

    def close(self) -> None:
        if self.stream.value:
            self._check(self.lib.aclrtDestroyStream(self.stream), "aclrtDestroyStream")
            self.stream = ctypes.c_void_p()
        if self.context.value:
            self._check(self.lib.aclrtDestroyContext(self.context), "aclrtDestroyContext")
            self.context = ctypes.c_void_p()
        self.lib.aclrtResetDevice(self.device)
        self.lib.aclFinalize()


@dataclass
class DeviceArray:
    runtime: AclRuntime
    host: np.ndarray
    pointer: ctypes.c_void_p
    acl_dtype: int | None = None

    @classmethod
    def from_host(
        cls, runtime: AclRuntime, value: np.ndarray, acl_dtype: int | None = None
    ) -> "DeviceArray":
        host = np.ascontiguousarray(value)
        pointer = runtime.malloc(host.nbytes)
        runtime.copy_to_device(pointer, host)
        return cls(runtime, host, pointer, acl_dtype)

    @classmethod
    def empty(
        cls, runtime: AclRuntime, shape: tuple[int, ...], dtype: np.dtype,
        acl_dtype: int | None = None,
    ) -> "DeviceArray":
        host = np.empty(shape, dtype=dtype)
        return cls(runtime, host, runtime.malloc(host.nbytes), acl_dtype)

    def fetch(self) -> np.ndarray:
        self.runtime.copy_to_host(self.host, self.pointer)
        return self.host.copy()

    def close(self) -> None:
        if self.pointer.value:
            self.runtime.free(self.pointer)
            self.pointer = ctypes.c_void_p()


@dataclass(frozen=True)
class Case:
    name: str
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    sparse_indices: np.ndarray
    actual_query: np.ndarray | None
    actual_kv: np.ndarray | None
    query_rope: np.ndarray
    key_rope: np.ndarray
    scale: float
    sparse_mode: int
    return_aux: bool
    primary_acl_dtype: int | None = None
    sparse_block_size: int = 1
    query_rope_acl_dtype: int | None = None
    key_rope_acl_dtype: int | None = None
    rope_omission_gap: float = 0.0


def float32_to_bf16_storage(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    output = (rounded >> np.uint32(16)).astype(np.uint16)
    nan_mask = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (bits & np.uint32(0x007FFFFF)) != 0
    )
    output[nan_mask] = ((bits[nan_mask] >> np.uint32(16)) | np.uint32(0x40)).astype(np.uint16)
    return np.ascontiguousarray(output)


def bf16_storage_to_float32(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def deterministic_inputs(
    qs: int,
    kvs: int,
    qn: int,
    dtype: np.dtype,
    *,
    b: int = 1,
    rope_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    content = ((np.arange(512, dtype=np.float32) % 17) - 8.0) / 64.0
    key_content = ((np.arange(512, dtype=np.float32) % 19) - 9.0) / 72.0
    query = np.empty((b, qs, qn, 512), dtype=np.float32)
    key = np.empty((b, kvs, 1, 512), dtype=np.float32)
    value = np.empty((b, kvs, 1, 512), dtype=np.float32)
    for batch in range(b):
        for q in range(qs):
            for h in range(qn):
                query[batch, q, h] = (content * (q + 1) * (h + 1) +
                                       (h - q + batch) / 128.0)
        for k in range(kvs):
            key[batch, k, 0] = key_content * (k + 1) + (k + batch) / 256.0
            value[batch, k, 0] = (k + 1 + batch) * 0.125 + content / 8.0

    rope_base = ((np.arange(64, dtype=np.float32) % 11) - 5.0) / 16.0
    query_rope = np.empty((b, qs, qn, 64), dtype=np.float32)
    key_rope = np.empty((b, kvs, 1, 64), dtype=np.float32)
    for batch in range(b):
        for q in range(qs):
            for h in range(qn):
                query_rope[batch, q, h] = rope_base * (q + 1) * (h + 1 + batch)
        for k in range(kvs):
            sign = -1.0 if (k + batch) % 2 else 1.0
            key_rope[batch, k, 0] = rope_base * sign * (k + 1)

    if rope_only:
        query.fill(0.0)
        key.fill(0.0)
    return tuple(np.ascontiguousarray(x, dtype=dtype) for x in (query, key, value, query_rope, key_rope))


def make_random_independent_case(*, wide: bool) -> Case:
    """Independent Q/K/V/RoPE coverage that cannot pass through K/V aliasing."""
    rng = np.random.default_rng(20260825 if wide else 20260824)
    if wide:
        query = rng.uniform(-10.0, 100.0, (1, 1, 1, 512)).astype(np.float16)
        key = rng.uniform(5.0, 100.0, (1, 32, 1, 512)).astype(np.float16)
        value = rng.uniform(-10.0, 100.0, (1, 32, 1, 512)).astype(np.float16)
        query_rope = rng.uniform(-10.0, 10.0, (1, 1, 1, 64)).astype(np.float16)
        key_rope = rng.uniform(-10.0, 10.0, (1, 32, 1, 64)).astype(np.float16)
        sparse_indices = np.asarray(
            [[[[0, 2, 5, 7, 11, 13, 17, 19, 23, 29, 30, 31]]]], dtype=np.int32
        )
        name = "L_random_wide"
        scale = 0.0884
    else:
        query = rng.normal(0.0, 0.2, (1, 3, 4, 512)).astype(np.float16)
        key = rng.normal(0.0, 0.2, (1, 11, 1, 512)).astype(np.float16)
        value = rng.uniform(-1.0, 1.0, (1, 11, 1, 512)).astype(np.float16)
        query_rope = rng.normal(0.0, 0.2, (1, 3, 4, 64)).astype(np.float16)
        key_rope = rng.normal(0.0, 0.2, (1, 11, 1, 64)).astype(np.float16)
        sparse_indices = np.asarray(
            [[[[0, 3, 6, 9, 10]], [[1, 2, 5, 7, 9]], [[0, 4, 6, 8, 10]]]],
            dtype=np.int32,
        )
        name = "L_random_diffuse"
        scale = 1.0 / np.sqrt(576.0)
    return Case(
        name=name,
        query=np.ascontiguousarray(query),
        key=np.ascontiguousarray(key),
        value=np.ascontiguousarray(value),
        sparse_indices=np.ascontiguousarray(sparse_indices),
        actual_query=None,
        actual_kv=None,
        query_rope=np.ascontiguousarray(query_rope),
        key_rope=np.ascontiguousarray(key_rope),
        scale=scale,
        sparse_mode=0,
        return_aux=True,
    )


def make_case(
    name: str,
    sparse_rows: list[list[int]],
    *,
    kvs: int,
    qn: int = 1,
    b: int = 1,
    dtype: np.dtype = np.dtype(np.float16),
    actual_query: int | None = None,
    actual_kv: int | None = None,
    scale: float = 0.125,
    sparse_mode: int = 0,
    sparse_block_size: int = 1,
    return_aux: bool = True,
    rope_only: bool = False,
    bf16: bool = False,
) -> Case:
    qs = len(sparse_rows)
    query, key, value, query_rope, key_rope = deterministic_inputs(
        qs, kvs, qn, dtype, b=b, rope_only=rope_only
    )
    if bf16:
        query, key, value, query_rope, key_rope = (
            float32_to_bf16_storage(x) for x in (query, key, value, query_rope, key_rope)
        )
    indices = np.asarray(sparse_rows, dtype=np.int32).reshape(1, qs, 1, -1)
    indices = np.repeat(indices, b, axis=0)
    actual_query_array = None if actual_query is None else np.full((b,), actual_query, dtype=np.int32)
    actual_kv_array = None if actual_kv is None else np.full((b,), actual_kv, dtype=np.int32)
    case = Case(
        name,
        query,
        key,
        value,
        indices,
        actual_query_array,
        actual_kv_array,
        query_rope,
        key_rope,
        scale,
        sparse_mode,
        return_aux,
        ACL_BF16 if bf16 else None,
        sparse_block_size,
    )
    if rope_only:
        expected, _, _ = cpu_reference(case)
        no_rope = Case(**{**case.__dict__, "query_rope": np.zeros_like(query_rope), "key_rope": np.zeros_like(key_rope)})
        omitted, _, _ = cpu_reference(no_rope)
        object.__setattr__(case, "rope_omission_gap", float(np.max(np.abs(expected.astype(np.float64) - omitted.astype(np.float64)))))
    return case


def make_content_only_single_index_case() -> Case:
    case = make_case(
        "L_content_single_index",
        [[2]],
        kvs=4,
        dtype=np.dtype(np.float16),
        return_aux=True,
    )
    return replace(
        case,
        query_rope=np.zeros_like(case.query_rope),
        key_rope=np.zeros_like(case.key_rope),
    )


def build_cases() -> list[Case]:
    fp16 = np.dtype(np.float16)
    fp32 = np.dtype(np.float32)
    return [
        make_case("A_basic", [[0, 1, 3], [2, 3, 1]], kvs=4, dtype=fp16),
        make_case("B_shared_kv_heads", [[0, 2, 3], [1, 3, 0]], kvs=4, qn=2, dtype=fp16),
        make_case("C_invalid_suffix", [[0, 2, -1], [3, 9, -1]], kvs=4, dtype=fp16),
        make_case("D_rope_required", [[0, 1, 2], [1, 2, 3]], kvs=4, dtype=fp16, rope_only=True),
        make_case(
            "D_rope_required_fp32",
            [[0, 1, 2], [1, 2, 3]],
            kvs=4,
            dtype=fp32,
            rope_only=True,
        ),
        make_case(
            "D_rope_required_bf16",
            [[0, 1, 2], [1, 2, 3]],
            kvs=4,
            dtype=fp32,
            rope_only=True,
            bf16=True,
        ),
        make_case("E_actual_query", [[0, 1], [1, 2], [2, 3]], kvs=4, dtype=fp16, actual_query=2),
        make_case("E_actual_kv", [[0, 4, -1], [2, 3, 4]], kvs=5, dtype=fp16, actual_kv=3),
        make_case(
            "E_actual_both",
            [[0, 3, -1], [1, 4, -1], [2, 3, -1]],
            kvs=5,
            dtype=fp16,
            actual_query=2,
            actual_kv=4,
        ),
        make_case(
            "F_right_down_causal",
            [[0, 3, 4], [1, 3, 4], [2, 3, 4]],
            kvs=5,
            dtype=fp16,
            sparse_mode=3,
        ),
        make_case(
            "G_aux_disabled",
            [[0, 1, 3], [2, 3, 1]],
            kvs=4,
            dtype=fp16,
            return_aux=False,
        ),
        make_case("H_fp32", [[0, 1, 3], [2, 3, 1]], kvs=4, dtype=fp32),
        make_case(
            "I_bf16",
            [[0, 1, 3], [2, 3, 1]],
            kvs=4,
            dtype=fp32,
            bf16=True,
        ),
        make_case(
            "J_actual_causal",
            [[0, 2, 3], [1, 2, 3], [0, 1, 2]],
            kvs=5,
            dtype=fp16,
            actual_query=2,
            actual_kv=4,
            sparse_mode=3,
        ),
        make_case(
            "L_rope_single_index",
            [[2]],
            kvs=4,
            dtype=fp16,
            rope_only=True,
            return_aux=True,
        ),
        make_content_only_single_index_case(),
        make_random_independent_case(wide=False),
    ]


def build_910b_launch_cases() -> list[Case]:
    """Nontrivial target-launch coverage outside the small regression set."""
    fp16 = np.dtype(np.float16)
    fp32 = np.dtype(np.float32)

    def rows(qs: int, indices: list[int]) -> list[list[int]]:
        return [indices[q % len(indices):] + indices[:q % len(indices)] for q in range(qs)]

    return [
        make_case("L_batch2_qn4", rows(8, [0, 2, 5]), kvs=16, qn=4, b=2, dtype=fp16),
        make_case("L_batch4_qn2", rows(4, [0, 3, 7]), kvs=16, qn=2, b=4, dtype=fp16),
        make_case("L_qn128", [[0, 3, 7]], kvs=8, qn=128, dtype=fp16),
        make_case(
            "L_long_kv128_sparse16",
            rows(16, list(range(0, 128, 8))),
            kvs=128,
            qn=8,
            dtype=fp16,
        ),
        make_case("L_block2_batch2", rows(8, [0, 3, 7]), kvs=16, qn=2, b=2,
                  dtype=fp16, sparse_block_size=2),
        make_case("L_block4_causal", rows(8, [0, 3, 7]), kvs=32, qn=2,
                  dtype=fp16, sparse_mode=3, sparse_block_size=4),
        make_case("L_block8", rows(8, [0, 3, 7]), kvs=64, qn=2,
                  dtype=fp32, sparse_block_size=8),
        make_case("L_block16_actual", rows(8, [0, 3, 5]), kvs=96, qn=2, b=2,
                  dtype=fp16, actual_query=6, actual_kv=96, sparse_mode=3,
                  sparse_block_size=16),
        make_case("L_block128", [[0, 1, 2]], kvs=384, qn=1,
                  dtype=fp16, sparse_block_size=128),
        make_case("L_batch2_bf16", rows(4, [0, 2, 5]), kvs=16, qn=4, b=2,
                  dtype=fp32, bf16=True),
        make_random_independent_case(wide=False),
        make_random_independent_case(wide=True),
    ]


def storage_for_dtype(shape: tuple[int, ...], dtype_name: str) -> tuple[np.ndarray, int | None]:
    if dtype_name == "float16":
        return np.zeros(shape, dtype=np.float16), None
    if dtype_name == "float32":
        return np.zeros(shape, dtype=np.float32), None
    if dtype_name == "bfloat16":
        return np.zeros(shape, dtype=np.uint16), ACL_BF16
    raise ValueError(f"unsupported matrix dtype: {dtype_name}")


def make_workspace_case(
    name: str,
    *,
    dtype_name: str = "float16",
    rope_dtype_name: str | None = None,
    b: int = 1,
    qs: int = 2,
    kvs: int = 5,
    qn: int = 1,
    sparse_size: int = 3,
    sparse_block_size: int = 1,
    sparse_mode: int = 0,
    actual_presence: str = "absent",
    return_aux: bool = False,
) -> Case:
    rope_dtype_name = dtype_name if rope_dtype_name is None else rope_dtype_name
    query, primary_acl_dtype = storage_for_dtype((b, qs, qn, 512), dtype_name)
    key, _ = storage_for_dtype((b, kvs, 1, 512), dtype_name)
    value, _ = storage_for_dtype((b, kvs, 1, 512), dtype_name)
    query_rope, query_rope_acl_dtype = storage_for_dtype((b, qs, qn, 64), rope_dtype_name)
    key_rope, key_rope_acl_dtype = storage_for_dtype((b, kvs, 1, 64), rope_dtype_name)
    sparse_indices = np.zeros((b, qs, 1, sparse_size), dtype=np.int32)
    actual_query = (
        np.full((b,), qs, dtype=np.int32)
        if actual_presence in ("query", "both") else None
    )
    actual_kv = (
        np.full((b,), kvs, dtype=np.int32)
        if actual_presence in ("kv", "both") else None
    )
    return Case(
        name=name,
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        actual_query=actual_query,
        actual_kv=actual_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=0.125,
        sparse_mode=sparse_mode,
        return_aux=return_aux,
        primary_acl_dtype=primary_acl_dtype,
        sparse_block_size=sparse_block_size,
        query_rope_acl_dtype=query_rope_acl_dtype,
        key_rope_acl_dtype=key_rope_acl_dtype,
    )


def build_workspace_cases() -> list[Case]:
    """One-axis legal-domain matrix; no case launches the device kernel."""
    cases: list[Case] = []

    def add(name: str, **overrides: object) -> None:
        cases.append(make_workspace_case(name, **overrides))

    for dtype_name in ("float16", "float32", "bfloat16"):
        add(f"dtype_{dtype_name}", dtype_name=dtype_name)
    for b in (1, 2):
        add(f"batch_{b}", b=b)
    for qn in (1, 2, 4, 8, 16, 32, 64, 128):
        add(f"query_heads_{qn}", qn=qn)
    for qs, kvs in ((1, 1), (1, 8), (8, 1), (2, 5), (5, 2), (16, 16)):
        add(f"seq_{qs}_{kvs}", qs=qs, kvs=kvs)
    for sparse_size, kvs, relation in (
        (1, 5, "one"),
        (2, 5, "two"),
        (3, 5, "three"),
        (4, 5, "less"),
        (5, 5, "equal"),
        (7, 5, "greater"),
    ):
        add(f"sparse_size_{relation}", sparse_size=sparse_size, kvs=kvs)
    for block_size in (1, 2, 4, 8, 128):
        add(f"sparse_block_{block_size}", sparse_block_size=block_size)
    for sparse_mode in (0, 3):
        add(f"sparse_mode_{sparse_mode}", sparse_mode=sparse_mode)
    for presence in ("absent", "query", "kv", "both"):
        add(f"actual_{presence}", actual_presence=presence)
    for return_aux in (False, True):
        add(f"return_lse_{str(return_aux).lower()}", return_aux=return_aux)
    # OpDef dtype/format lists are index-paired. The generated simplified keys
    # therefore admit exactly these three same-dtype combinations.
    for dtype_name in ("float16", "float32", "bfloat16"):
        add(f"rope_{dtype_name}_{dtype_name}", dtype_name=dtype_name,
            rope_dtype_name=dtype_name)
    return cases


def cpu_reference(case: Case) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Contract-level reference using gather, vector dot products, and batch softmax."""
    bsz, qs, qn, _ = case.query.shape
    kvs = case.key.shape[1]
    output = np.zeros(case.query.shape, dtype=np.float64)
    max_output = np.full((bsz, 1, qs, qn), EMPTY_MAX, dtype=np.float64)
    sum_output = np.zeros((bsz, 1, qs, qn), dtype=np.float64)
    scale = float(np.float16(case.scale))
    scale_experiment = os.environ.get("SFA_REFERENCE_SCALE", "attribute")
    if scale_experiment == "half_attribute":
        scale *= 0.5
    elif scale_experiment == "inv_sqrt_512":
        scale = 1.0 / np.sqrt(512.0)
    elif scale_experiment == "inv_sqrt_576":
        scale = 1.0 / np.sqrt(576.0)

    for b in range(bsz):
        query_len = qs if case.actual_query is None else min(max(int(case.actual_query[b]), 0), qs)
        kv_len = kvs if case.actual_kv is None else min(max(int(case.actual_kv[b]), 0), kvs)
        for q in range(query_len):
            positions: list[int] = []
            for sparse_block_index in case.sparse_indices[b, q, 0].astype(np.int64):
                # The operator consumes block indices; a negative entry is an
                # invalid suffix sentinel, matching the device loop.
                if sparse_block_index < 0:
                    break
                block_start = int(sparse_block_index) * int(case.sparse_block_size)
                if block_start >= kv_len:
                    continue
                for block_offset in range(case.sparse_block_size):
                    key_pos = block_start + block_offset
                    if key_pos >= kv_len:
                        break
                    positions.append(key_pos)
            if case.sparse_mode == 3:
                causal_experiment = os.environ.get("SFA_REFERENCE_CAUSAL", "right_down_actual")
                if causal_experiment == "ordinary":
                    causal_limit = q
                elif causal_experiment == "right_down_physical":
                    causal_limit = q + kvs - qs
                else:
                    causal_limit = q + kv_len - query_len
                positions = [pos for pos in positions if pos <= causal_limit]
            if not positions:
                continue
            positions = np.asarray(positions, dtype=np.int64)
            convert = bf16_storage_to_float32 if case.primary_acl_dtype == ACL_BF16 else lambda x: x
            selected_key = convert(case.key[b, positions, 0]).astype(np.float64)
            selected_rope = convert(case.key_rope[b, positions, 0]).astype(np.float64)
            aggregate_key = os.environ.get("SFA_REFERENCE_AGGREGATION", "value") == "key"
            selected_value = convert(
                case.key[b, positions, 0] if aggregate_key else case.value[b, positions, 0]
            ).astype(np.float64)
            query = convert(case.query[b, q]).astype(np.float64)
            query_rope = convert(case.query_rope[b, q]).astype(np.float64)
            content_scores = query @ selected_key.T
            rope_scores = (
                np.zeros((qn, selected_rope.shape[0]), dtype=np.float64)
                if os.environ.get("SFA_REFERENCE_ROPE", "enabled") == "disabled"
                else query_rope @ selected_rope.T
            )
            if os.environ.get("SFA_REFERENCE_ROPE_SCALE", "scaled") == "unscaled":
                scores = content_scores * scale + rope_scores
            else:
                scores = (content_scores + rope_scores) * scale
            row_max = scores.max(axis=1, keepdims=True)
            exponentials = np.exp(scores - row_max)
            row_sum = exponentials.sum(axis=1, keepdims=True)
            output[b, q] = (exponentials / row_sum) @ selected_value
            max_output[b, 0, q] = row_max[:, 0]
            sum_output[b, 0, q] = row_sum[:, 0]

    output = (float32_to_bf16_storage(output) if case.primary_acl_dtype == ACL_BF16
              else output.astype(case.query.dtype))
    if not case.return_aux:
        return output, None, None
    return output, max_output.astype(np.float32), sum_output.astype(np.float32)


class SparseFlashAttentionAclnn:
    def __init__(self, runtime: AclRuntime) -> None:
        library_path = os.environ.get("SPARSE_FLASH_ATTENTION_CUSTOM_LIB")
        if not library_path or not os.path.isfile(library_path):
            raise RuntimeError("SPARSE_FLASH_ATTENTION_CUSTOM_LIB must point to the generated shared library")
        tiling_library_path = os.environ.get("SPARSE_FLASH_ATTENTION_TILING_LIB")
        if tiling_library_path and not os.path.isfile(tiling_library_path):
            raise RuntimeError(
                "SPARSE_FLASH_ATTENTION_TILING_LIB must point to the generated tiling shared library"
            )
        self.runtime = runtime
        self.nnopbase = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
        # A packaged custom OPP loads the Host tiling library through its
        # registry. Direct ACLNN tests load loose build artifacts, so retain an
        # explicit handle when supplied; otherwise 561002 can occur before the
        # Kernel is selected even though libcust_opapi itself loaded correctly.
        self.tiling = (
            None
            if not tiling_library_path
            else ctypes.CDLL(tiling_library_path, mode=ctypes.RTLD_GLOBAL)
        )
        if tiling_library_path:
            print(
                "TILING_SHARED_LIBRARY_LOAD_PASS=" + os.path.realpath(tiling_library_path),
                flush=True,
            )
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
        self.custom.aclnnSparseFlashAttentionGetWorkspaceSize.argtypes = [
            *([ctypes.c_void_p] * 8),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.custom.aclnnSparseFlashAttentionGetWorkspaceSize.restype = ctypes.c_int
        self.custom.aclnnSparseFlashAttention.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.custom.aclnnSparseFlashAttention.restype = ctypes.c_int
        self.last_device_ms: float | None = None

    def _tensor(self, array: DeviceArray) -> ctypes.c_void_p:
        shape = tuple(array.host.shape)
        dims = (ctypes.c_int64 * len(shape))(*shape)
        strides_list = []
        stride = 1
        for dimension in reversed(shape):
            strides_list.append(stride)
            stride *= dimension
        strides = (ctypes.c_int64 * len(shape))(*reversed(strides_list))
        handle = self.nnopbase.aclCreateTensor(
            dims,
            len(shape),
            array.acl_dtype if array.acl_dtype is not None else dtype_code(array.host),
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

    def run(self, case: Case) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        input_arrays = [
            DeviceArray.from_host(self.runtime, value, case.primary_acl_dtype)
            for value in (case.query, case.key, case.value)
        ]
        input_arrays.append(DeviceArray.from_host(self.runtime, case.sparse_indices))
        optional_arrays = [
            None if value is None else DeviceArray.from_host(self.runtime, value)
            for value in (case.actual_query, case.actual_kv)
        ]
        rope_arrays = [
            DeviceArray.from_host(
                self.runtime, case.query_rope,
                case.primary_acl_dtype if case.query_rope_acl_dtype is None
                else case.query_rope_acl_dtype,
            ),
            DeviceArray.from_host(
                self.runtime, case.key_rope,
                case.primary_acl_dtype if case.key_rope_acl_dtype is None
                else case.key_rope_acl_dtype,
            ),
        ]
        attention = DeviceArray.empty(
            self.runtime, tuple(case.query.shape), case.query.dtype, case.primary_acl_dtype
        )
        aux_shape = (case.query.shape[0], 1, case.query.shape[1], case.query.shape[2])
        max_output = DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32)) if case.return_aux else None
        sum_output = DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32)) if case.return_aux else None
        all_arrays = input_arrays + [x for x in optional_arrays if x is not None] + rope_arrays + [attention]
        all_arrays += [x for x in (max_output, sum_output) if x is not None]
        handles: list[ctypes.c_void_p] = []
        workspace = None
        start_event = ctypes.c_void_p()
        end_event = ctypes.c_void_p()
        try:
            input_handles = [self._tensor(value) for value in input_arrays]
            handles.extend(input_handles)
            optional_handles = []
            for value in optional_arrays:
                handle = ctypes.c_void_p() if value is None else self._tensor(value)
                optional_handles.append(handle)
                if value is not None:
                    handles.append(handle)
            rope_handles = [self._tensor(value) for value in rope_arrays]
            handles.extend(rope_handles)
            attention_handle = self._tensor(attention)
            handles.append(attention_handle)
            max_handle = ctypes.c_void_p() if max_output is None else self._tensor(max_output)
            sum_handle = ctypes.c_void_p() if sum_output is None else self._tensor(sum_output)
            if max_output is not None:
                handles.append(max_handle)
                handles.append(sum_handle)

            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            status = self.custom.aclnnSparseFlashAttentionGetWorkspaceSize(
                *input_handles,
                *optional_handles,
                *rope_handles,
                case.scale,
                case.sparse_block_size,
                case.sparse_mode,
                2,
                case.return_aux,
                attention_handle,
                max_handle,
                sum_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"aclnnSparseFlashAttentionGetWorkspaceSize failed: {status}")
            print(f"WORKSPACE_SIZE case={case.name} bytes={workspace_size.value}", flush=True)
            workspace_pointer = ctypes.c_void_p()
            if workspace_size.value:
                workspace = self.runtime.malloc(workspace_size.value)
                workspace_pointer = workspace
            self.runtime._check(
                self.runtime.lib.aclrtCreateEvent(ctypes.byref(start_event)),
                "aclrtCreateEvent(start)",
            )
            self.runtime._check(
                self.runtime.lib.aclrtCreateEvent(ctypes.byref(end_event)),
                "aclrtCreateEvent(end)",
            )
            self.runtime._check(
                self.runtime.lib.aclrtRecordEvent(start_event, self.runtime.stream),
                "aclrtRecordEvent(start)",
            )
            status = self.custom.aclnnSparseFlashAttention(
                workspace_pointer,
                workspace_size.value,
                executor,
                self.runtime.stream,
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"aclnnSparseFlashAttention failed: {status}")
            self.runtime._check(
                self.runtime.lib.aclrtRecordEvent(end_event, self.runtime.stream),
                "aclrtRecordEvent(end)",
            )
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeEvent(end_event),
                "aclrtSynchronizeEvent(end)",
            )
            elapsed_ms = ctypes.c_float()
            self.runtime._check(
                self.runtime.lib.aclrtEventElapsedTime(
                    ctypes.byref(elapsed_ms), start_event, end_event
                ),
                "aclrtEventElapsedTime",
            )
            self.last_device_ms = float(elapsed_ms.value)
            return (
                attention.fetch(),
                None if max_output is None else max_output.fetch(),
                None if sum_output is None else sum_output.fetch(),
            )
        finally:
            if end_event.value:
                self.runtime.lib.aclrtDestroyEvent(end_event)
            if start_event.value:
                self.runtime.lib.aclrtDestroyEvent(start_event)
            if workspace is not None:
                self.runtime.free(workspace)
            for handle in handles:
                self.nnopbase.aclDestroyTensor(handle)
            for array in all_arrays:
                array.close()

    def get_workspace_size(self, case: Case) -> dict[str, object]:
        input_arrays = [
            DeviceArray.from_host(self.runtime, value, case.primary_acl_dtype)
            for value in (case.query, case.key, case.value)
        ]
        input_arrays.append(DeviceArray.from_host(self.runtime, case.sparse_indices))
        optional_arrays = [
            None if value is None else DeviceArray.from_host(self.runtime, value)
            for value in (case.actual_query, case.actual_kv)
        ]
        rope_arrays = [
            DeviceArray.from_host(
                self.runtime, case.query_rope,
                case.primary_acl_dtype if case.query_rope_acl_dtype is None
                else case.query_rope_acl_dtype,
            ),
            DeviceArray.from_host(
                self.runtime, case.key_rope,
                case.primary_acl_dtype if case.key_rope_acl_dtype is None
                else case.key_rope_acl_dtype,
            ),
        ]
        attention = DeviceArray.empty(
            self.runtime, tuple(case.query.shape), case.query.dtype, case.primary_acl_dtype
        )
        aux_shape = (case.query.shape[0], 1, case.query.shape[1], case.query.shape[2])
        max_output = DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32)) if case.return_aux else None
        sum_output = DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32)) if case.return_aux else None
        all_arrays = input_arrays + [x for x in optional_arrays if x is not None] + rope_arrays + [attention]
        all_arrays += [x for x in (max_output, sum_output) if x is not None]
        handles: list[ctypes.c_void_p] = []
        try:
            input_handles = [self._tensor(value) for value in input_arrays]
            handles.extend(input_handles)
            optional_handles = []
            for value in optional_arrays:
                handle = ctypes.c_void_p() if value is None else self._tensor(value)
                optional_handles.append(handle)
                if value is not None:
                    handles.append(handle)
            rope_handles = [self._tensor(value) for value in rope_arrays]
            handles.extend(rope_handles)
            attention_handle = self._tensor(attention)
            handles.append(attention_handle)
            max_handle = ctypes.c_void_p() if max_output is None else self._tensor(max_output)
            sum_handle = ctypes.c_void_p() if sum_output is None else self._tensor(sum_output)
            if max_output is not None:
                handles.extend((max_handle, sum_handle))

            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            status = self.custom.aclnnSparseFlashAttentionGetWorkspaceSize(
                *input_handles,
                *optional_handles,
                *rope_handles,
                case.scale,
                case.sparse_block_size,
                case.sparse_mode,
                2,
                case.return_aux,
                attention_handle,
                max_handle,
                sum_handle,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            result = {
                "case": case.name,
                "return_code": int(status),
                "workspace_size": int(workspace_size.value),
                "executor_created": bool(executor.value),
                "is_561002": status == 561002,
                "shapes": {
                    "query": list(case.query.shape),
                    "key": list(case.key.shape),
                    "value": list(case.value.shape),
                    "sparse_indices": list(case.sparse_indices.shape),
                    "query_rope": list(case.query_rope.shape),
                    "key_rope": list(case.key_rope.shape),
                },
                "dtypes": {
                    "query_key_value": acl_dtype_name(case.primary_acl_dtype, case.query.dtype),
                    "query_rope": acl_dtype_name(
                        case.primary_acl_dtype if case.query_rope_acl_dtype is None
                        else case.query_rope_acl_dtype,
                        case.query_rope.dtype,
                    ),
                    "key_rope": acl_dtype_name(
                        case.primary_acl_dtype if case.key_rope_acl_dtype is None
                        else case.key_rope_acl_dtype,
                        case.key_rope.dtype,
                    ),
                    "sparse_indices": "int32",
                },
                "attrs": {
                    "scale_value": case.scale,
                    "sparse_block_size": case.sparse_block_size,
                    "sparse_mode": case.sparse_mode,
                    "attention_mode": 2,
                    "return_softmax_lse": case.return_aux,
                },
                "actual_seq_lengths": {
                    "query_present": case.actual_query is not None,
                    "kv_present": case.actual_kv is not None,
                },
            }
            print(json.dumps(result, sort_keys=True), flush=True)
            return result
        finally:
            for handle in handles:
                self.nnopbase.aclDestroyTensor(handle)
            for array in all_arrays:
                array.close()


def error_metrics(
    actual: np.ndarray, expected: np.ndarray, acl_dtype: int | None = None
) -> tuple[float, float]:
    if acl_dtype == ACL_BF16:
        actual = bf16_storage_to_float32(actual)
        expected = bf16_storage_to_float32(expected)
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    absolute = np.abs(actual64 - expected64)
    meaningful = np.abs(expected64) > 1.0e-8
    relative = absolute[meaningful] / np.abs(expected64[meaningful]) if np.any(meaningful) else np.asarray([0.0])
    return float(absolute.max(initial=0.0)), float(relative.max(initial=0.0))


def acl_dtype_name(override: int | None, storage_dtype: np.dtype) -> str:
    if override == ACL_BF16:
        return "bfloat16"
    return str(storage_dtype)


def run_case(
    op: SparseFlashAttentionAclnn,
    case: Case,
    dump_output_dir: Path | None = None,
) -> dict[str, object]:
    expected_attention, expected_max, expected_sum = cpu_reference(case)
    actual_attention, actual_max, actual_sum = op.run(case)
    if dump_output_dir is not None:
        dump_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(dump_output_dir / f"{case.name}-attention.npy", actual_attention)
        if actual_max is not None:
            np.save(dump_output_dir / f"{case.name}-softmax-max.npy", actual_max)
            np.save(dump_output_dir / f"{case.name}-softmax-sum.npy", actual_sum)
    attention_abs, attention_rel = error_metrics(
        actual_attention, expected_attention, case.primary_acl_dtype
    )
    max_abs = None if actual_max is None else error_metrics(actual_max, expected_max)[0]
    sum_abs = None if actual_sum is None else error_metrics(actual_sum, expected_sum)[0]
    has_nonfinite = not np.isfinite(actual_attention).all()
    if actual_max is not None:
        has_nonfinite |= not np.isfinite(actual_max).all() or not np.isfinite(actual_sum).all()
    if case.primary_acl_dtype == ACL_BF16:
        actual_compare = bf16_storage_to_float32(actual_attention)
        expected_compare = bf16_storage_to_float32(expected_attention)
        attention_ok = np.allclose(actual_compare, expected_compare, rtol=4.0e-3, atol=4.0e-3)
    else:
        attention_ok = np.allclose(actual_attention, expected_attention, rtol=2.0e-3, atol=2.0e-3)
    aux_ok = True
    if actual_max is not None:
        aux_ok = np.allclose(actual_max, expected_max, rtol=1.0e-4, atol=1.0e-4)
        aux_ok &= np.allclose(actual_sum, expected_sum, rtol=1.0e-4, atol=1.0e-4)
    result = {
        "case": case.name,
        "shapes": {
            "query": list(case.query.shape),
            "key": list(case.key.shape),
            "value": list(case.value.shape),
            "sparse_indices": list(case.sparse_indices.shape),
            "actual_seq_lengths_query": None if case.actual_query is None else list(case.actual_query.shape),
            "actual_seq_lengths_kv": None if case.actual_kv is None else list(case.actual_kv.shape),
            "query_rope": list(case.query_rope.shape),
            "key_rope": list(case.key_rope.shape),
        },
        "dtype": "bfloat16" if case.primary_acl_dtype == ACL_BF16 else str(case.query.dtype),
        "attrs": {
            "scale_value": case.scale,
            "sparse_block_size": case.sparse_block_size,
            "sparse_mode": case.sparse_mode,
            "attention_mode": 2,
            "return_softmax_lse": case.return_aux,
        },
        "sparse_indices": case.sparse_indices.tolist(),
        "actual_lengths": {
            "query": None if case.actual_query is None else case.actual_query.tolist(),
            "kv": None if case.actual_kv is None else case.actual_kv.tolist(),
        },
        "attention_max_abs_error": attention_abs,
        "attention_max_rel_error": attention_rel,
        "softmax_max_max_abs_error": max_abs,
        "softmax_sum_max_abs_error": sum_abs,
        "softmax_max_actual": None if actual_max is None else actual_max.tolist(),
        "softmax_max_expected": None if expected_max is None else expected_max.tolist(),
        "softmax_sum_actual": None if actual_sum is None else actual_sum.tolist(),
        "softmax_sum_expected": None if expected_sum is None else expected_sum.tolist(),
        "rope_omission_detection_gap": case.rope_omission_gap or None,
        "actual_has_nan_or_inf": bool(has_nonfinite),
        "status": "PASS" if attention_ok and aux_ok and not has_nonfinite else "FAIL",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="all", help="case name or all")
    parser.add_argument("--device", type=int, default=int(os.environ.get("SPARSE_FLASH_ATTENTION_DEVICE_ID", "4")))
    parser.add_argument(
        "--workspace-only-matrix",
        action="store_true",
        help="call GetWorkspaceSize over the legal-domain matrix without launching a kernel",
    )
    parser.add_argument(
        "--workspace-only",
        action="store_true",
        help="call GetWorkspaceSize for the selected regular correctness case without launching a kernel",
    )
    parser.add_argument(
        "--launch-matrix",
        action="store_true",
        help="run the nontrivial 910B target-launch correctness matrix",
    )
    parser.add_argument(
        "--dump-output-dir",
        type=Path,
        help="save direct D2H attention/max/sum arrays for device differential probes",
    )
    args = parser.parse_args()
    cases = (build_workspace_cases() if args.workspace_only_matrix else
             build_910b_launch_cases() if args.launch_matrix else build_cases())
    selected = cases if args.case == "all" else [case for case in cases if case.name == args.case]
    if not selected:
        raise SystemExit(f"unknown case {args.case}; choices: {', '.join(case.name for case in cases)}")
    print("CAUSAL_PREDICATE=key_position <= query_position + actual_kv_length - actual_query_length", flush=True)
    runtime = AclRuntime(args.device)
    try:
        op = SparseFlashAttentionAclnn(runtime)
        if args.workspace_only_matrix or args.workspace_only:
            results = [op.get_workspace_size(case) for case in selected]
        else:
            results = [run_case(op, case, args.dump_output_dir) for case in selected]
    finally:
        runtime.close()
    if args.workspace_only_matrix or args.workspace_only:
        passed = sum(
            result["return_code"] == ACL_SUCCESS and result["executor_created"]
            for result in results
        )
    else:
        passed = sum(result["status"] == "PASS" for result in results)
    print(f"SUMMARY passed={passed} total={len(results)}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Differential probes against the installed official SparseFlashAttention ACLNN.

This test is intentionally independent from the competition source.  It loads
the official CANN ``libopapi.so`` and the generated custom ACLNN library as
separate handles, invokes both on identical device tensors, and compares each
result with a NumPy reference.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE_PATH = pathlib.Path(__file__).with_name("test_sparse_flash_attention.py")
SPEC = importlib.util.spec_from_file_location("sfa_local_runner", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import local runner: {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


ACL_SUCCESS = BASE.ACL_SUCCESS
INT64_MAX = (1 << 63) - 1


# This is the exact CANN 8.5 header contract shipped with the isolated
# environment used by this test.  Keep this table next to the ctypes ABI
# declaration so an accidental argument shift is visible in the audit log.
HEADER_PARAMS = [
    (1, "query", "const aclTensor *"),
    (2, "key", "const aclTensor *"),
    (3, "value", "const aclTensor *"),
    (4, "sparseIndices", "const aclTensor *"),
    (5, "blockTableOptional", "const aclTensor *"),
    (6, "actualSeqLengthsQueryOptional", "const aclTensor *"),
    (7, "actualSeqLengthsKvOptional", "const aclTensor *"),
    (8, "queryRopeOptional", "const aclTensor *"),
    (9, "keyRopeOptional", "const aclTensor *"),
    (10, "scaleValue", "double"),
    (11, "sparseBlockSize", "int64_t"),
    (12, "layoutQueryOptional", "char *"),
    (13, "layoutKvOptional", "char *"),
    (14, "sparseMode", "int64_t"),
    (15, "preTokens", "int64_t"),
    (16, "nextTokens", "int64_t"),
    (17, "attentionMode", "int64_t"),
    (18, "returnSoftmaxLse", "bool"),
    (19, "attentionOutOut", "const aclTensor *"),
    (20, "softmaxMaxOut", "const aclTensor *"),
    (21, "softmaxSumOut", "const aclTensor *"),
    (22, "workspaceSize", "uint64_t *"),
    (23, "executor", "aclOpExecutor **"),
]


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _symbol_owner(function: Any) -> tuple[int, str]:
    """Return function address and the shared object reported by dladdr."""
    address = ctypes.cast(function, ctypes.c_void_p).value or 0
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if not libdl.dladdr(ctypes.c_void_p(address), ctypes.byref(info)):
        return address, "<dladdr failed>"
    return address, (info.dli_fname.decode() if info.dli_fname else "<unknown>")


def _load_library(path: str, mode: int = ctypes.RTLD_GLOBAL) -> ctypes.CDLL:
    return ctypes.CDLL(path, mode=mode)


class OfficialSparseFlashAttention:
    """Thin adapter for the complete official CANN ACLNN signature."""

    def __init__(self, runtime: BASE.AclRuntime, library_path: str) -> None:
        self.runtime = runtime
        # aclCreateTensor lives in nnopbase, and must be globally visible to
        # both official and custom opapi libraries.
        self.nnopbase = _load_library("libnnopbase.so")
        self.official = _load_library(library_path)
        self.nnopbase.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,
        ]
        self.nnopbase.aclCreateTensor.restype = ctypes.c_void_p
        self.nnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyTensor.restype = ctypes.c_int
        self.official.aclnnSparseFlashAttentionGetWorkspaceSize.argtypes = [
            *([ctypes.c_void_p] * 9),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 3),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.official.aclnnSparseFlashAttentionGetWorkspaceSize.restype = ctypes.c_int
        self.official.aclnnSparseFlashAttention.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.official.aclnnSparseFlashAttention.restype = ctypes.c_int
        self._signature_printed = False
        address, owner = _symbol_owner(self.official.aclnnSparseFlashAttentionGetWorkspaceSize)
        expected = os.path.realpath(library_path)
        print(f"SYMBOL name=aclnnSparseFlashAttentionGetWorkspaceSize address=0x{address:x} owner={owner}", flush=True)
        print(f"SYMBOL_EXPECTED_OWNER={expected} match={os.path.realpath(owner) == expected}", flush=True)
        address, owner = _symbol_owner(self.official.aclnnSparseFlashAttention)
        print(f"SYMBOL name=aclnnSparseFlashAttention address=0x{address:x} owner={owner}", flush=True)
        print(f"SYMBOL_EXPECTED_OWNER={expected} match={os.path.realpath(owner) == expected}", flush=True)

    @staticmethod
    def _ctype_name(value: Any) -> str:
        if value is ctypes.c_void_p:
            return "ctypes.c_void_p"
        if value is ctypes.c_double:
            return "ctypes.c_double"
        if value is ctypes.c_int64:
            return "ctypes.c_int64"
        if value is ctypes.c_char_p:
            return "ctypes.c_char_p"
        if value is ctypes.c_bool:
            return "ctypes.c_bool"
        if value is ctypes.POINTER(ctypes.c_uint64):
            return "ctypes.POINTER(ctypes.c_uint64)"
        if value is ctypes.POINTER(ctypes.c_void_p):
            return "ctypes.POINTER(ctypes.c_void_p)"
        return repr(value)

    def _print_signature(self, actual_labels: list[str]) -> None:
        if self._signature_printed:
            return
        print("LOCAL_HEADER_HAS blockTable=YES sinks=NO layoutQuery=YES layoutKv=YES preTokens=YES nextTokens=YES", flush=True)
        argtypes = self.official.aclnnSparseFlashAttentionGetWorkspaceSize.argtypes
        print("ABI_TABLE index | C header | ctypes argtype | Python actual argument", flush=True)
        for (index, name, c_type), c_arg, actual in zip(HEADER_PARAMS, argtypes, actual_labels):
            print(f"ABI_TABLE {index:02d} {name} | {c_type} | {self._ctype_name(c_arg)} | {actual}", flush=True)
        if len(argtypes) != len(HEADER_PARAMS) or len(actual_labels) != len(HEADER_PARAMS):
            raise AssertionError("GetWorkspaceSize ABI parameter count mismatch")
        self._signature_printed = True

    def _gws_args(
        self,
        input_handles: list[ctypes.c_void_p],
        actual_handles: list[ctypes.c_void_p],
        rope_handles: list[ctypes.c_void_p],
        output_handles: list[ctypes.c_void_p],
        case: BASE.Case,
        workspace_size: Any,
        executor: Any,
    ) -> list[Any]:
        args = [
            input_handles[0], input_handles[1], input_handles[2], input_handles[3],
            ctypes.c_void_p(), actual_handles[0], actual_handles[1],
            rope_handles[0], rope_handles[1],
            float(case.scale), int(case.sparse_block_size), b"BSND", b"BSND",
            int(case.sparse_mode), INT64_MAX, INT64_MAX, 2, True,
            output_handles[0], output_handles[1], output_handles[2],
            ctypes.byref(workspace_size), ctypes.byref(executor),
        ]
        actual_labels = [
            "query_handle", "key_handle", "value_handle", "sparse_indices_handle",
            "NULL(blockTableOptional)",
            "actual_query_handle" if actual_handles[0].value else "NULL(actual_query)",
            "actual_kv_handle" if actual_handles[1].value else "NULL(actual_kv)",
            "query_rope_handle", "key_rope_handle", "float(scale)", "int(sparse_block_size)",
            "b'BSND'", "b'BSND'", "int(sparse_mode)", "INT64_MAX", "INT64_MAX",
            "2", "True", "attention_handle", "softmax_max_handle", "softmax_sum_handle",
            "byref(c_uint64 workspace_size)", "byref(c_void_p executor)",
        ]
        self._print_signature(actual_labels)
        return args

    @staticmethod
    def _ptr(value: BASE.DeviceArray) -> str:
        return f"0x{int(value.pointer.value if hasattr(value.pointer, 'value') else value.pointer):x}"

    @staticmethod
    def _direct_fetch(runtime: BASE.AclRuntime, value: BASE.DeviceArray) -> np.ndarray:
        result = np.empty_like(value.host)
        runtime.copy_to_host(result, value.pointer)
        return result

    def _tensor(self, array: BASE.DeviceArray) -> ctypes.c_void_p:
        shape = tuple(array.host.shape)
        dims = (ctypes.c_int64 * len(shape))(*shape)
        strides_list: list[int] = []
        stride = 1
        for dimension in reversed(shape):
            strides_list.append(stride)
            stride *= dimension
        strides = (ctypes.c_int64 * len(shape))(*reversed(strides_list))
        handle = self.nnopbase.aclCreateTensor(
            dims,
            len(shape),
            array.acl_dtype if array.acl_dtype is not None else BASE.dtype_code(array.host),
            strides,
            0,
            BASE.ACL_FORMAT_ND,
            dims,
            len(shape),
            array.pointer,
        )
        if not handle:
            raise RuntimeError("aclCreateTensor returned null")
        return ctypes.c_void_p(handle)

    def run(self, case: BASE.Case) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        input_arrays = [
            BASE.DeviceArray.from_host(self.runtime, value, case.primary_acl_dtype)
            for value in (case.query, case.key, case.value, case.sparse_indices)
        ]
        actual_arrays = [
            None if value is None else BASE.DeviceArray.from_host(self.runtime, value)
            for value in (case.actual_query, case.actual_kv)
        ]
        rope_arrays = [
            BASE.DeviceArray.from_host(self.runtime, case.query_rope, case.query_rope_acl_dtype),
            BASE.DeviceArray.from_host(self.runtime, case.key_rope, case.key_rope_acl_dtype),
        ]
        attention = BASE.DeviceArray.empty(
            self.runtime, tuple(case.query.shape), case.query.dtype, case.primary_acl_dtype
        )
        aux_shape = (case.query.shape[0], 1, case.query.shape[1], case.query.shape[2])
        max_output = BASE.DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32))
        sum_output = BASE.DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32))
        all_arrays = input_arrays + [x for x in actual_arrays if x is not None]
        all_arrays += rope_arrays + [attention, max_output, sum_output]
        handles: list[ctypes.c_void_p] = []
        workspace = None
        try:
            input_handles = [self._tensor(value) for value in input_arrays]
            handles.extend(input_handles)
            actual_handles: list[ctypes.c_void_p] = []
            for value in actual_arrays:
                handle = ctypes.c_void_p() if value is None else self._tensor(value)
                actual_handles.append(handle)
                if value is not None:
                    handles.append(handle)
            rope_handles = [self._tensor(value) for value in rope_arrays]
            handles.extend(rope_handles)
            output_handles = [self._tensor(value) for value in (attention, max_output, sum_output)]
            handles.extend(output_handles)
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            args = self._gws_args(
                input_handles, actual_handles, rope_handles, output_handles,
                case, workspace_size, executor,
            )
            status = self.official.aclnnSparseFlashAttentionGetWorkspaceSize(*args)
            if status != ACL_SUCCESS:
                raise RuntimeError(f"official GetWorkspaceSize failed: {status}")
            workspace_pointer = ctypes.c_void_p()
            if workspace_size.value:
                workspace = self.runtime.malloc(workspace_size.value)
                workspace_pointer = workspace
            status = self.official.aclnnSparseFlashAttention(
                workspace_pointer, workspace_size.value, executor, self.runtime.stream,
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"official execution failed: {status}")
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeStream(self.runtime.stream),
                "aclrtSynchronizeStream(official)",
            )
            return attention.fetch(), max_output.fetch(), sum_output.fetch()
        finally:
            if workspace is not None:
                self.runtime.free(workspace)
            for handle in handles:
                self.nnopbase.aclDestroyTensor(handle)
            for array in all_arrays:
                array.close()

    def audit_run(self, case: BASE.Case, output_dir: str) -> None:
        """Run one official case with explicit handles and direct D2H fetches."""
        os.makedirs(output_dir, exist_ok=True)
        input_arrays = [
            BASE.DeviceArray.from_host(self.runtime, value, case.primary_acl_dtype)
            for value in (case.query, case.key, case.value, case.sparse_indices)
        ]
        key_before = case.key.copy()
        value_before = case.value.copy()
        actual_arrays = [
            None if value is None else BASE.DeviceArray.from_host(self.runtime, value)
            for value in (case.actual_query, case.actual_kv)
        ]
        rope_arrays = [
            BASE.DeviceArray.from_host(self.runtime, case.query_rope, case.query_rope_acl_dtype),
            BASE.DeviceArray.from_host(self.runtime, case.key_rope, case.key_rope_acl_dtype),
        ]
        attention = BASE.DeviceArray.empty(
            self.runtime, tuple(case.query.shape), case.query.dtype, case.primary_acl_dtype
        )
        aux_shape = (case.query.shape[0], 1, case.query.shape[1], case.query.shape[2])
        max_output = BASE.DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32))
        sum_output = BASE.DeviceArray.empty(self.runtime, aux_shape, np.dtype(np.float32))
        all_arrays = input_arrays + [x for x in actual_arrays if x is not None]
        all_arrays += rope_arrays + [attention, max_output, sum_output]
        handles: list[ctypes.c_void_p] = []
        workspace = None
        try:
            names = ("query", "key", "value", "sparse_indices")
            input_handles = [self._tensor(value) for value in input_arrays]
            handles.extend(input_handles)
            actual_handles: list[ctypes.c_void_p] = []
            for value in actual_arrays:
                handle = ctypes.c_void_p() if value is None else self._tensor(value)
                actual_handles.append(handle)
                if value is not None:
                    handles.append(handle)
            rope_handles = [self._tensor(value) for value in rope_arrays]
            handles.extend(rope_handles)
            output_values = (attention, max_output, sum_output)
            output_names = ("attentionOut", "softmaxMax", "softmaxSum")
            output_handles = [self._tensor(value) for value in output_values]
            handles.extend(output_handles)
            for name, array in zip(names, input_arrays):
                print(f"AUDIT_PTR {case.name} {name} device={self._ptr(array)}", flush=True)
            for name, array, handle in zip(names, input_arrays, input_handles):
                print(f"AUDIT_HANDLE {case.name} {name} aclTensor=0x{int(handle.value):x} device={self._ptr(array)}", flush=True)
            for name, array, handle in zip(output_names, output_values, output_handles):
                print(f"AUDIT_PTR {case.name} {name} device={self._ptr(array)}", flush=True)
                print(f"AUDIT_HANDLE {case.name} {name} aclTensor=0x{int(handle.value):x} device={self._ptr(array)}", flush=True)
            if any(attention.pointer.value == value.pointer.value for value in input_arrays[:3]):
                raise AssertionError("attentionOut device pointer aliases query/key/value")
            if any(output_handles[0].value == handle.value for handle in input_handles[:3]):
                raise AssertionError("attentionOut aclTensor handle aliases query/key/value")
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            args = self._gws_args(
                input_handles, actual_handles, rope_handles, output_handles,
                case, workspace_size, executor,
            )
            print(f"AUDIT_CALL {case.name} attention_arg_is_output={args[18] is output_handles[0]}", flush=True)
            status = self.official.aclnnSparseFlashAttentionGetWorkspaceSize(*args)
            print(f"AUDIT_GWS {case.name} status={status} workspace={workspace_size.value} executor=0x{int(executor.value or 0):x}", flush=True)
            if status != ACL_SUCCESS:
                raise RuntimeError(f"official GetWorkspaceSize failed: {status}")
            workspace_pointer = ctypes.c_void_p()
            if workspace_size.value:
                workspace = self.runtime.malloc(workspace_size.value)
                workspace_pointer = workspace
            status = self.official.aclnnSparseFlashAttention(
                workspace_pointer, workspace_size.value, executor, self.runtime.stream,
            )
            if status != ACL_SUCCESS:
                raise RuntimeError(f"official execution failed: {status}")
            self.runtime._check(
                self.runtime.lib.aclrtSynchronizeStream(self.runtime.stream),
                "aclrtSynchronizeStream(official audit)",
            )
            key_after = self._direct_fetch(self.runtime, input_arrays[1])
            print(f"AUDIT_D2H {case.name} key", flush=True)
            value_after = self._direct_fetch(self.runtime, input_arrays[2])
            print(f"AUDIT_D2H {case.name} value", flush=True)
            attention_after = self._direct_fetch(self.runtime, attention)
            print(f"AUDIT_D2H {case.name} attention", flush=True)
            max_after = self._direct_fetch(self.runtime, max_output)
            sum_after = self._direct_fetch(self.runtime, sum_output)
            np.save(os.path.join(output_dir, "official_key.npy"), key_after)
            np.save(os.path.join(output_dir, "official_value.npy"), value_after)
            np.save(os.path.join(output_dir, "official_attention.npy"), attention_after)
            np.save(os.path.join(output_dir, "official_softmax_max.npy"), max_after)
            np.save(os.path.join(output_dir, "official_softmax_sum.npy"), sum_after)
            keyf = BASE.bf16_storage_to_float32(key_after) if key_after.dtype == np.uint16 else key_after.astype(np.float32)
            valuef = BASE.bf16_storage_to_float32(value_after) if value_after.dtype == np.uint16 else value_after.astype(np.float32)
            attentionf = BASE.bf16_storage_to_float32(attention_after) if attention_after.dtype == np.uint16 else attention_after.astype(np.float32)
            # The sparse index is 3 in both constant fingerprints.  Compare
            # the output against the selected key/value row, not the entire
            # [S_k] tensor (which would broadcast or compare incompatible
            # shapes and hide the actual fingerprint).
            selected_key = keyf[:, 3:4, :, :]
            selected_value = valuef[:, 3:4, :, :]
            print(f"AUDIT_MUTATION {case.name} key_max_abs={float(np.max(np.abs(key_after.astype(np.float64) - key_before.astype(np.float64))))} value_max_abs={float(np.max(np.abs(value_after.astype(np.float64) - value_before.astype(np.float64))))}", flush=True)
            print(f"AUDIT_DISTANCE {case.name} attention_selected_key_max_abs={float(np.max(np.abs(attentionf.astype(np.float64) - selected_key.astype(np.float64))))} attention_selected_value_max_abs={float(np.max(np.abs(attentionf.astype(np.float64) - selected_value.astype(np.float64))))}", flush=True)
            key_sample = keyf[0, 3, 0, :8] if keyf.ndim == 4 and keyf.shape[1] > 3 else keyf.reshape(-1)[:8]
            value_sample = valuef[0, 3, 0, :8] if valuef.ndim == 4 and valuef.shape[1] > 3 else valuef.reshape(-1)[:8]
            attention_sample = attentionf.reshape(-1)[:8]
            print(f"AUDIT_SAMPLE {case.name} key[3,0:8]={key_sample.tolist()} value[3,0:8]={value_sample.tolist()} attention[0:8]={attention_sample.tolist()}", flush=True)
        finally:
            if workspace is not None:
                self.runtime.free(workspace)
            for handle in handles:
                self.nnopbase.aclDestroyTensor(handle)
            for array in all_arrays:
                array.close()


def _metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    a = BASE.bf16_storage_to_float32(actual) if actual.dtype == np.uint16 else actual
    e = BASE.bf16_storage_to_float32(expected) if expected.dtype == np.uint16 else expected
    delta = np.abs(a.astype(np.float64) - e.astype(np.float64))
    denom = np.abs(e.astype(np.float64))
    mask = denom > 1.0e-8
    rel = delta[mask] / denom[mask] if np.any(mask) else np.asarray([0.0])
    return float(delta.max(initial=0.0)), float(rel.max(initial=0.0))


def _first_diff(official: np.ndarray, custom: np.ndarray, numpy_ref: np.ndarray) -> dict[str, Any]:
    of = BASE.bf16_storage_to_float32(official) if official.dtype == np.uint16 else official
    cu = BASE.bf16_storage_to_float32(custom) if custom.dtype == np.uint16 else custom
    nr = BASE.bf16_storage_to_float32(numpy_ref) if numpy_ref.dtype == np.uint16 else numpy_ref
    delta = np.abs(cu.astype(np.float64) - of.astype(np.float64))
    index = tuple(int(x) for x in np.unravel_index(int(np.argmax(delta)), delta.shape))
    return {
        "index": index,
        "official_value": float(of[index]),
        "custom_value": float(cu[index]),
        "numpy_value": float(nr[index]),
        "custom_official": _metrics(custom, official),
        "numpy_official": _metrics(numpy_ref, official),
        "custom_numpy_max_abs": float(np.max(np.abs(cu.astype(np.float64) - nr.astype(np.float64)))),
    }


def _run_probe(
    official: OfficialSparseFlashAttention,
    custom: BASE.SparseFlashAttentionAclnn,
    case: BASE.Case,
) -> dict[str, Any]:
    numpy_ref, _, _ = BASE.cpu_reference(case)
    official_out, official_max, official_sum = official.run(case)
    custom_out, custom_max, custom_sum = custom.run(case)
    result: dict[str, Any] = {"probe": case.name, "attention": _first_diff(official_out, custom_out, numpy_ref)}
    result["softmax_max_official_custom"] = _metrics(custom_max, official_max)
    result["softmax_sum_official_custom"] = _metrics(custom_sum, official_sum)
    result["softmax_max_numpy_official"] = _metrics(
        np.asarray(BASE.cpu_reference(case)[1]), official_max
    )
    result["softmax_sum_numpy_official"] = _metrics(
        np.asarray(BASE.cpu_reference(case)[2]), official_sum
    )
    result["status"] = "PASS" if result["attention"]["custom_official"] == (0.0, 0.0) else "DIVERGED"
    print(result, flush=True)
    return result


def build_probes() -> list[BASE.Case]:
    fp16 = np.dtype(np.float16)
    # Probe 1: exactly one valid sparse index; output must be V[index].
    query, key, value, qr, kr = BASE.deterministic_inputs(1, 8, 1, fp16)
    p1 = BASE.Case("probe1_single_index", query, key, value, np.asarray([[[[3]]]], np.int32),
                   None, None, qr, kr, 1.0, 0, True)
    # Probe 2: zero Q/K/RoPE gives equal scores and an arithmetic V mean.
    query, key, value, qr, kr = BASE.deterministic_inputs(1, 8, 1, fp16)
    query.fill(0); key.fill(0); qr.fill(0); kr.fill(0)
    p2 = BASE.Case("probe2_equal_score", query, key, value, np.asarray([[[[2, 5]]]], np.int32),
                   None, None, qr, kr, 1.0, 0, True)
    # Probe 3: content-only.
    query, key, value, qr, kr = BASE.deterministic_inputs(1, 8, 1, fp16)
    qr.fill(0); kr.fill(0)
    p3 = BASE.Case("probe3_content_only", query, key, value, np.asarray([[[[1, 3, 6]]]], np.int32),
                   None, None, qr, kr, 0.125, 0, True)
    # Probe 4: RoPE-only.
    query, key, value, qr, kr = BASE.deterministic_inputs(1, 8, 1, fp16, rope_only=True)
    p4 = BASE.Case("probe4_rope_only", query, key, value, np.asarray([[[[1, 3, 6]]]], np.int32),
                   None, None, qr, kr, 0.125, 0, True)
    # Probe 5: combined content and RoPE.
    query, key, value, qr, kr = BASE.deterministic_inputs(1, 8, 1, fp16)
    p5 = BASE.Case("probe5_combined", query, key, value, np.asarray([[[[1, 3, 6]]]], np.int32),
                   None, None, qr, kr, 0.125, 0, True)
    # Probe 6: same content-only input at several scales.
    scales = (1.0, 0.5, 0.125, 0.0884)
    probes = [p1, p2, p3, p4, p5]
    for index, scale in enumerate(scales):
        probes.append(BASE.Case(
            f"probe6_scale_{scale}", p3.query.copy(), p3.key.copy(), p3.value.copy(),
            p3.sparse_indices.copy(), None, None, p3.query_rope.copy(), p3.key_rope.copy(),
            scale, 0, True,
        ))
    return probes


def build_audit_cases() -> list[BASE.Case]:
    """Two independent constant fingerprints; no variable swapping is used."""
    fp16 = np.dtype(np.float16)
    query = np.zeros((1, 1, 1, 512), dtype=fp16)
    qr = np.zeros((1, 1, 1, 64), dtype=fp16)
    kr = np.zeros((1, 8, 1, 64), dtype=fp16)
    indices = np.asarray([[[[3]]]], dtype=np.int32)
    cases: list[BASE.Case] = []
    for name, key_value, value_value in (("constant_key7_value9", 7, 9), ("constant_key13_value29", 13, 29)):
        cases.append(BASE.Case(
            name,
            query.copy(),
            np.full((1, 8, 1, 512), key_value, dtype=fp16),
            np.full((1, 8, 1, 512), value_value, dtype=fp16),
            indices.copy(), None, None, qr.copy(), kr.copy(), 1.0, 0, True,
        ))
    return cases


def _run_audit_side(side: str, cases: list[BASE.Case], output_root: str, official_lib: str | None) -> int:
    runtime = BASE.AclRuntime(int(os.environ.get("SPARSE_FLASH_ATTENTION_DEVICE_ID", "4")))
    try:
        if side != "official":
            raise RuntimeError("the direct-buffer audit is only valid for the official side")
        if not official_lib:
            raise RuntimeError("official library path is required")
        implementation = OfficialSparseFlashAttention(runtime, official_lib)
        for case in cases:
            implementation.audit_run(case, os.path.join(output_root, case.name))
            print(f"OFFICIAL_AUDIT_PASS={case.name}", flush=True)
    finally:
        runtime.close()
    return 0


def _run_diff_side(
    side: str, cases: list[BASE.Case], output_root: str, official_lib: str | None,
) -> int:
    """Run only the requested two differential cases, using individual .npy files."""
    runtime = BASE.AclRuntime(int(os.environ.get("SPARSE_FLASH_ATTENTION_DEVICE_ID", "4")))
    try:
        if side == "official":
            if not official_lib:
                raise RuntimeError("official library path is required")
            implementation: Any = OfficialSparseFlashAttention(runtime, official_lib)
        else:
            implementation = BASE.SparseFlashAttentionAclnn(runtime)
        for case in cases:
            case_dir = os.path.join(output_root, case.name)
            os.makedirs(case_dir, exist_ok=True)
            if side == "official":
                implementation.audit_run(case, case_dir)
            else:
                output, max_output, sum_output = implementation.run(case)
                np.save(os.path.join(case_dir, "custom_attention.npy"), output)
                np.save(os.path.join(case_dir, "custom_softmax_max.npy"), max_output)
                np.save(os.path.join(case_dir, "custom_softmax_sum.npy"), sum_output)
            print(f"{side.upper()}_DIFF_PASS={case.name}", flush=True)
    finally:
        runtime.close()
    return 0


def _load_explicit(path: str) -> np.ndarray:
    """Explicit comparator load; intentionally no npz/list/tuple indirection."""
    return np.load(path, allow_pickle=False)


def run_audit_workflow(args: argparse.Namespace) -> int:
    root = args.output_dir or tempfile.mkdtemp(prefix="sfa-official-audit-")
    os.makedirs(root, exist_ok=True)
    print(f"AUDIT_OUTPUT_DIR={root}", flush=True)
    env = os.environ.copy()
    audit_cases = build_audit_cases()
    audit_root = os.path.join(root, "constants")
    command = [
        sys.executable, str(pathlib.Path(__file__).resolve()), "--device", str(args.device),
        "--audit-side", "official", "--output-dir", audit_root,
        "--official-lib", args.official_lib, "--custom-lib", args.custom_lib,
    ]
    subprocess.run(command, check=True, env=env)
    for case in audit_cases:
        case_dir = os.path.join(audit_root, case.name)
        key = _load_explicit(os.path.join(case_dir, "official_key.npy"))
        value = _load_explicit(os.path.join(case_dir, "official_value.npy"))
        attention = _load_explicit(os.path.join(case_dir, "official_attention.npy"))
        keyf = BASE.bf16_storage_to_float32(key) if key.dtype == np.uint16 else key.astype(np.float32)
        valuef = BASE.bf16_storage_to_float32(value) if value.dtype == np.uint16 else value.astype(np.float32)
        attentionf = BASE.bf16_storage_to_float32(attention) if attention.dtype == np.uint16 else attention.astype(np.float32)
        selected_key = keyf[:, 3:4, :, :]
        selected_value = valuef[:, 3:4, :, :]
        print(f"FINGERPRINT {case.name} attention_equals_selected_key={bool(np.array_equal(attentionf, selected_key))} attention_equals_selected_value={bool(np.array_equal(attentionf, selected_value))}", flush=True)

    diff_cases = build_probes()[:2]
    diff_root = os.path.join(root, "differential")
    for side in ("official", "custom"):
        command = [
            sys.executable, str(pathlib.Path(__file__).resolve()), "--device", str(args.device),
            "--diff-side", side, "--output-dir", os.path.join(diff_root, side),
            "--official-lib", args.official_lib, "--custom-lib", args.custom_lib,
        ]
        subprocess.run(command, check=True, env=env)
    for case in diff_cases:
        case_dir = os.path.join(diff_root, "official", case.name)
        custom_dir = os.path.join(diff_root, "custom", case.name)
        official_attention = _load_explicit(os.path.join(case_dir, "official_attention.npy"))
        custom_attention = _load_explicit(os.path.join(custom_dir, "custom_attention.npy"))
        numpy_attention, _, _ = BASE.cpu_reference(case)
        result = _first_diff(official_attention, custom_attention, numpy_attention)
        print({"probe": case.name, "attention": result}, flush=True)
    print("AUDIT_WORKFLOW_COMPLETE only=single-index,equal-score", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=int(os.environ.get("SPARSE_FLASH_ATTENTION_DEVICE_ID", "4")))
    parser.add_argument("--official-lib", default=os.environ.get("SPARSE_FLASH_ATTENTION_OFFICIAL_LIB"))
    parser.add_argument("--custom-lib", default=os.environ.get("SPARSE_FLASH_ATTENTION_CUSTOM_LIB"))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-side", choices=("official",))
    parser.add_argument("--diff-side", choices=("official", "custom"))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if not args.official_lib:
        raise SystemExit("--official-lib or SPARSE_FLASH_ATTENTION_OFFICIAL_LIB is required")
    if not args.custom_lib:
        raise SystemExit("--custom-lib or SPARSE_FLASH_ATTENTION_CUSTOM_LIB is required")
    os.environ["SPARSE_FLASH_ATTENTION_DEVICE_ID"] = str(args.device)
    if args.custom_lib:
        os.environ["SPARSE_FLASH_ATTENTION_CUSTOM_LIB"] = args.custom_lib
    if args.audit:
        return run_audit_workflow(args)
    if args.audit_side:
        return _run_audit_side(args.audit_side, build_audit_cases(), args.output_dir or ".", args.official_lib)
    if args.diff_side:
        return _run_diff_side(args.diff_side, build_probes()[:2], args.output_dir or ".", args.official_lib)
    # The default workflow is deliberately the restricted audit: constants,
    # then only single-index and equal-score differential cases.  There is no
    # npz/list-unpacking path, so every comparator input is an explicit .npy.
    return run_audit_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Minimal ACL lifecycle probe with per-step return codes and timings."""

from __future__ import annotations

import ctypes
import os
import time
from argparse import ArgumentParser
from pathlib import Path


ACL_SUCCESS = 0
KEY_LIBRARIES = ("libascendcl.so", "libascend_hal.so")


def timed_call(name: str, function, *args) -> int:
    start = time.monotonic()
    print(f"{name}: start_monotonic_s={start:.9f}", flush=True)
    result = int(function(*args))
    end = time.monotonic()
    elapsed_ms = (end - start) * 1000.0
    print(
        f"{name}: end_monotonic_s={end:.9f} elapsed_ms={elapsed_ms:.3f} "
        f"rc={result}",
        flush=True,
    )
    return result


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("MHC_EXPAND_DEVICE_ID", "0")),
        help="ACL device id (default: MHC_EXPAND_DEVICE_ID or 0)",
    )
    return parser.parse_args()


def print_loaded_libraries() -> None:
    paths: dict[str, set[str]] = {name: set() for name in KEY_LIBRARIES}
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        path = line.rsplit(maxsplit=1)[-1]
        for name in KEY_LIBRARIES:
            if Path(path).name.startswith(name):
                paths[name].add(str(Path(path).resolve()))
    for name in KEY_LIBRARIES:
        resolved = sorted(paths[name])
        print(
            f"loaded_library {name}={','.join(resolved) if resolved else 'NOT_LOADED'}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclFinalize.argtypes = []
    acl.aclFinalize.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtResetDevice.restype = ctypes.c_int
    acl.aclrtCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
    acl.aclrtCreateContext.restype = ctypes.c_int
    acl.aclrtDestroyContext.argtypes = [ctypes.c_void_p]
    acl.aclrtDestroyContext.restype = ctypes.c_int

    device = args.device
    print(f"device_id={device}", flush=True)
    context = ctypes.c_void_p()
    initialized = False
    device_set = False
    context_created = False
    failed = False

    try:
        rc = timed_call("aclInit", acl.aclInit, None)
        initialized = rc == ACL_SUCCESS
        if not initialized:
            return 1
        print_loaded_libraries()

        rc = timed_call("aclrtSetDevice", acl.aclrtSetDevice, device)
        device_set = rc == ACL_SUCCESS
        if not device_set:
            return 1

        rc = timed_call(
            "aclrtCreateContext", acl.aclrtCreateContext, ctypes.byref(context), device
        )
        context_created = rc == ACL_SUCCESS
        failed = not context_created
    finally:
        if context_created:
            failed |= timed_call(
                "aclrtDestroyContext", acl.aclrtDestroyContext, context
            ) != ACL_SUCCESS
        if device_set:
            failed |= timed_call(
                "aclrtResetDevice", acl.aclrtResetDevice, device
            ) != ACL_SUCCESS
        if initialized:
            failed |= timed_call("aclFinalize", acl.aclFinalize) != ACL_SUCCESS

    if failed:
        return 1
    print("ACL_RUNTIME_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

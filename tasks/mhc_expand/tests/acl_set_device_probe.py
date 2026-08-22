#!/usr/bin/env python3
"""Minimal ACL set-device lifecycle probe for runtime stability tests."""

from __future__ import annotations

import ctypes
import os
import time
from argparse import ArgumentParser


ACL_SUCCESS = 0


def timed_call(name: str, function, *args) -> int:
    start = time.monotonic()
    print(f"{name}: start_monotonic_s={start:.9f}", flush=True)
    result = int(function(*args))
    end = time.monotonic()
    print(
        f"{name}: end_monotonic_s={end:.9f} "
        f"elapsed_ms={(end - start) * 1000.0:.3f} rc={result}",
        flush=True,
    )
    return result


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("MHC_EXPAND_DEVICE_ID", "0")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtResetDevice.restype = ctypes.c_int
    acl.aclFinalize.argtypes = []
    acl.aclFinalize.restype = ctypes.c_int

    print(f"pid={os.getpid()} device_id={args.device}", flush=True)
    initialized = False
    device_set = False
    failed = False
    try:
        init_rc = timed_call("aclInit", acl.aclInit, None)
        initialized = init_rc == ACL_SUCCESS
        if not initialized:
            return 1

        set_rc = timed_call("aclrtSetDevice", acl.aclrtSetDevice, args.device)
        device_set = set_rc == ACL_SUCCESS
        if not device_set:
            return 1
    finally:
        if device_set:
            failed |= (
                timed_call("aclrtResetDevice", acl.aclrtResetDevice, args.device)
                != ACL_SUCCESS
            )
        if initialized:
            failed |= timed_call("aclFinalize", acl.aclFinalize) != ACL_SUCCESS

    if failed:
        return 1
    print("ACL_SET_DEVICE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Public CANNJudge package identity helpers.

No authentication or submission is performed here. The official problem package
is hashed so platform evidence can be bound to the exact template/contract
version that was current when the evidence was produced.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

DEFAULT_BASE_URL = "https://cannjudge.cn"


def official_package_sha256(problem_id: str, *, base_url: str | None = None, timeout: int = 30) -> str:
    problem_id = (problem_id or "").strip()
    if not problem_id:
        raise ValueError("missing CANNJUDGE_PROBLEM_ID")
    base = (base_url or os.environ.get("CANNJUDGE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = "%s/api/problems/%s/package" % (base, problem_id)
    request = urllib.request.Request(url, headers={"User-Agent": "xingchenbei-harness/1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package_sha(env: dict, *, timeout: int = 30) -> str:
    observed = official_package_sha256(
        env.get("CANNJUDGE_PROBLEM_ID", ""),
        base_url=env.get("CANNJUDGE_BASE_URL"),
        timeout=timeout,
    )
    expected = (env.get("CANNJUDGE_PACKAGE_SHA256") or "").strip().lower()
    if expected and expected != observed.lower():
        raise RuntimeError(
            "official package SHA mismatch: expected=%s observed=%s" % (expected, observed)
        )
    return observed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="hash the current official CANNJudge problem package")
    parser.add_argument("problem_id")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    print(official_package_sha256(args.problem_id, base_url=args.base_url))

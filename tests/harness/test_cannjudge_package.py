#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import io
import pathlib


def load_module(path):
    spec = importlib.util.spec_from_file_location("cannjudge_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, data):
        self.stream = io.BytesIO(data)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, size=-1):
        return self.stream.read(size)


def main():
    src = pathlib.Path(__file__).resolve().parents[2] / "tools" / "cannjudge_package.py"
    mod = load_module(src)
    payload = b"official-package-v1"
    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append((request.full_url, timeout))
        return Response(payload)

    mod.urllib.request.urlopen = fake_urlopen
    expected = hashlib.sha256(payload).hexdigest()
    observed = mod.official_package_sha256("problem-1", base_url="https://example.invalid")
    assert observed == expected
    assert calls[0][0] == "https://example.invalid/api/problems/problem-1/package"

    env = {
        "CANNJUDGE_PROBLEM_ID": "problem-1",
        "CANNJUDGE_BASE_URL": "https://example.invalid",
        "CANNJUDGE_PACKAGE_SHA256": expected,
    }
    assert mod.resolve_package_sha(env) == expected

    env["CANNJUDGE_PACKAGE_SHA256"] = "0" * 64
    try:
        mod.resolve_package_sha(env)
    except RuntimeError as exc:
        assert "package SHA mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatch to fail")

    print("PASS cannjudge_package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

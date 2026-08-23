#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile


def load_module(path):
    spec = importlib.util.spec_from_file_location("task_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    src = pathlib.Path(__file__).resolve().parents[2] / "tools" / "task_state.py"
    mod = load_module(src)
    root = pathlib.Path(tempfile.mkdtemp(prefix="task-state-test-"))
    runs = root / "tasks" / "demo" / "runs" / "harness"
    runs.mkdir(parents=True)
    record = {
        "mode": "candidate",
        "name": "c1",
        "decision": "promote:local_improved",
        "score": 1.23,
        "direction": "lower",
        "stages": [
            {"stage": "build", "returncode": 0},
            {"stage": "validate", "returncode": 0},
            {"stage": "bench", "returncode": 0},
        ],
    }
    (runs / "20260823T000000Z-c1.json").write_text(json.dumps(record), encoding="utf-8")
    (runs / "best-local.json").write_text(json.dumps({"score": 1.23, "name": "c1"}), encoding="utf-8")

    mod.ROOT = root
    mod.git = lambda *args: {
        ("branch", "--show-current"): "task/demo",
        ("rev-parse", "--short", "HEAD"): "abcdef0",
        ("status", "--short"): " M x\n",
    }.get(tuple(args), "")

    data = mod.summarize("demo")
    assert data["latest"]["mode"] == "candidate"
    assert data["latest"]["stages"] == {"build": "pass", "validate": "pass", "bench": "pass"}
    assert data["best_local"]["score"] == 1.23
    assert data["dirty_paths"] == 1
    print("PASS task_state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

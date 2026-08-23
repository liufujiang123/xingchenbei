#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "tools"))
    fp = load_module(repo / "tools" / "evidence_fingerprint.py", "evidence_fingerprint")
    sys.modules["evidence_fingerprint"] = fp
    state = load_module(repo / "tools" / "task_state.py", "task_state")

    root = pathlib.Path(tempfile.mkdtemp(prefix="task-state-test-"))
    task = root / "tasks" / "demo"
    workspace = task / "workspace"
    scripts = task / "scripts"
    tests = task / "tests"
    runs = task / "runs" / "harness"
    config = root / "config" / "tasks" / "demo.env"
    for path in (workspace, scripts, tests, runs, config.parent):
        path.mkdir(parents=True, exist_ok=True)
    (workspace / "kernel.cpp").write_text("v1\n", encoding="utf-8")
    (scripts / "build.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    (scripts / "validate.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    (scripts / "bench.sh").write_text("#!/bin/sh\necho SCORE=1\n", encoding="utf-8")
    (tests / "cases.json").write_text('{"case":1}\n', encoding="utf-8")
    config.write_text(
        "TASK_NAME=demo\n"
        "TASK_DIR=tasks/demo\n"
        "WORKSPACE_DIR=tasks/demo/workspace\n"
        "BUILD_CMD='tasks/demo/scripts/build.sh'\n"
        "VALIDATE_CMD='tasks/demo/scripts/validate.sh'\n"
        "BENCH_CMD='tasks/demo/scripts/bench.sh'\n"
        "CANN_VERSION=8.5\n"
        "TARGET_SOC=ascend910b\n",
        encoding="utf-8",
    )

    fp.ROOT = root
    env = fp.load_env(config)
    recorded = fp.capture("demo", config_path=config, env=env)
    record_path = runs / "20260823T000000Z-c1.json"
    record_path.write_text(json.dumps({
        "mode": "candidate",
        "name": "c1",
        "decision": "promote:local_improved",
        "score": 1.23,
        "direction": "lower",
        "stages": [
            {"stage": "guard", "returncode": 0, "output": fp.marker(recorded) + "\n"},
            {"stage": "build", "returncode": 0},
            {"stage": "validate", "returncode": 0},
            {"stage": "bench", "returncode": 0},
        ],
    }), encoding="utf-8")
    (runs / "best-local.json").write_text(json.dumps({
        "run": str(record_path.relative_to(root)), "score": 1.23, "name": "c1"
    }), encoding="utf-8")

    state.ROOT = root
    fp.ROOT = root
    state.git = lambda *args: {
        ("branch", "--show-current"): "task/demo",
        ("rev-parse", "--short", "HEAD"): "abcdef0",
        ("status", "--short"): "",
    }.get(tuple(args), "")

    data = state.summarize("demo")
    assert data["latest"]["freshness"]["status"] == "fresh"
    assert data["best_local"]["comparability"]["status"] == "compatible"

    (workspace / "kernel.cpp").write_text("v2\n", encoding="utf-8")
    data = state.summarize("demo")
    assert data["latest"]["freshness"]["status"] == "stale"
    assert data["best_local"]["freshness"]["status"] == "stale"
    assert data["best_local"]["comparability"]["status"] == "compatible"

    (tests / "cases.json").write_text('{"case":2}\n', encoding="utf-8")
    data = state.summarize("demo")
    assert data["best_local"]["comparability"]["status"] == "incompatible"

    legacy_path = runs / "20260823T010000Z-legacy.json"
    legacy_path.write_text(json.dumps({
        "mode": "validate", "decision": "passed",
        "stages": [{"stage": "validate", "returncode": 0}],
    }), encoding="utf-8")
    data = state.summarize("demo")
    assert data["latest"]["freshness"]["status"] == "unknown"

    print("PASS task_state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile


def load_module(path):
    spec = importlib.util.spec_from_file_location("evidence_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_task(root):
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
    (scripts / "bench.sh").write_text("#!/bin/sh\necho SCORE=1\n", encoding="utf-8")
    (tests / "cases.json").write_text('{"case":1}\n', encoding="utf-8")
    config.write_text(
        "TASK_NAME=demo\n"
        "TASK_DIR=tasks/demo\n"
        "WORKSPACE_DIR=tasks/demo/workspace\n"
        "BUILD_CMD='tasks/demo/scripts/build.sh'\n"
        "BENCH_CMD='tasks/demo/scripts/bench.sh'\n"
        "BENCH_DIRECTION=lower\n"
        "CANN_VERSION=8.5\n"
        "TARGET_SOC=ascend910b\n"
        "CANNJUDGE_PUBLIC_PROBLEM_ID=999\n"
        "CANNJUDGE_PROBLEM_ID=demo-internal\n",
        encoding="utf-8",
    )
    return task, workspace, scripts, tests, runs, config


def main():
    src = pathlib.Path(__file__).resolve().parents[2] / "tools" / "evidence_fingerprint.py"
    mod = load_module(src)
    root = pathlib.Path(tempfile.mkdtemp(prefix="evidence-fingerprint-test-"))
    task, workspace, scripts, tests, runs, config = make_task(root)
    mod.ROOT = root

    env = mod.load_env(config)
    fp1 = mod.capture("demo", config_path=config, env=env)
    encoded = mod.marker(fp1)
    assert mod.parse_marker(encoded) == mod.compact(fp1)

    # Irrelevant config text does not reset the benchmark context when the
    # resolved commands/environment/cases are unchanged.
    config.write_text(config.read_text(encoding="utf-8") + "# comment only\n", encoding="utf-8")
    env_same = mod.load_env(config)
    fp_comment = mod.capture("demo", config_path=config, env=env_same)
    assert mod.score_compatibility(fp1, fp_comment, "bench")["status"] == "compatible"

    # Source changes invalidate proof for the current subject but are expected
    # between candidates, so same-context benchmark scores remain comparable.
    (workspace / "kernel.cpp").write_text("v2\n", encoding="utf-8")
    fp2 = mod.capture("demo", config_path=config, env=env)
    assert mod.stage_freshness(fp1, fp2, "bench")["status"] == "stale"
    assert "subject_changed" in mod.stage_freshness(fp1, fp2, "bench")["reasons"]
    assert mod.score_compatibility(fp1, fp2, "bench")["status"] == "compatible"

    # Case-set changes invalidate numerical comparison.
    (tests / "cases.json").write_text('{"case":2}\n', encoding="utf-8")
    fp3 = mod.capture("demo", config_path=config, env=env)
    cmp3 = mod.score_compatibility(fp2, fp3, "bench")
    assert cmp3["status"] == "incompatible"
    assert "case_set_changed" in cmp3["reasons"]

    # A compatible best is kept.
    run = runs / "20260823T000000Z-c1.json"
    run.write_text(json.dumps({
        "stages": [{"stage": "guard", "returncode": 0, "output": mod.marker(fp3) + "\n"}]
    }), encoding="utf-8")
    best = runs / "best-local.json"
    best.write_text(json.dumps({
        "run": str(run.relative_to(root)), "score": 1.0, "direction": "lower"
    }), encoding="utf-8")
    mod.prepare_best("demo", fp3)
    assert best.exists()

    # A new case context archives the old best so agent_loop cannot claim a
    # faster/slower result against an incomparable baseline.
    (tests / "cases.json").write_text('{"case":3}\n', encoding="utf-8")
    fp4 = mod.capture("demo", config_path=config, env=env)
    mod.prepare_best("demo", fp4)
    assert not best.exists()
    assert list(runs.glob("best-local.incompatible*.json"))

    # Legacy best without a fingerprint is also reset rather than silently used.
    best.write_text(json.dumps({"score": 3.0}), encoding="utf-8")
    mod.prepare_best("demo", fp4)
    assert not best.exists()
    assert list(runs.glob("best-local.unknown*.json"))

    print("PASS evidence_fingerprint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compact task/run state so Codex does not reread long evidence logs."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_record(task_dir: pathlib.Path):
    runs = task_dir / "runs" / "harness"
    if not runs.exists():
        return None, None
    records = sorted(
        p for p in runs.glob("*.json")
        if p.name not in ("best-local.json", "best-platform.json", "state.json", "context.json")
    )
    if not records:
        return None, None
    path = records[-1]
    return path, load_json(path)


def stage_summary(record):
    out = {}
    for stage in (record or {}).get("stages", []):
        name = stage.get("stage")
        if name:
            out[name] = "pass" if stage.get("returncode") == 0 else "fail"
    return out


def summarize(task: str) -> dict:
    task_dir = ROOT / "tasks" / task
    runs = task_dir / "runs" / "harness"
    latest_path, latest = latest_record(task_dir)
    best_local = load_json(runs / "best-local.json") if runs.exists() else None
    best_platform = load_json(runs / "best-platform.json") if runs.exists() else None

    return {
        "task": task,
        "branch": git("branch", "--show-current") or None,
        "head": git("rev-parse", "--short", "HEAD") or None,
        "dirty_paths": len([x for x in git("status", "--short").splitlines() if x.strip()]),
        "latest": None if latest is None else {
            "record": str(latest_path.relative_to(ROOT)),
            "mode": latest.get("mode"),
            "name": latest.get("name"),
            "decision": latest.get("decision"),
            "score": latest.get("score"),
            "direction": latest.get("direction"),
            "platform": latest.get("platform"),
            "stages": stage_summary(latest),
        },
        "best_local": best_local,
        "best_platform": best_platform,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="show compact harness task state")
    parser.add_argument("--task", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    data = summarize(args.task)
    if args.as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print("TASK=%s branch=%s head=%s dirty_paths=%d" % (
        data["task"], data["branch"] or "unknown", data["head"] or "unknown", data["dirty_paths"]
    ))
    latest = data["latest"]
    if latest:
        print("LATEST mode=%s decision=%s score=%s record=%s" % (
            latest.get("mode"), latest.get("decision"), latest.get("score"), latest.get("record")
        ))
        if latest.get("stages"):
            print("STAGES " + " ".join("%s=%s" % x for x in latest["stages"].items()))
        platform = latest.get("platform")
        if platform:
            print("PLATFORM status=%s score=%s submission_id=%s" % (
                platform.get("status"), platform.get("score"), platform.get("submission_id")
            ))
    else:
        print("LATEST=NONE")
    for key in ("best_local", "best_platform"):
        item = data[key]
        if item:
            print("%s score=%s name=%s" % (key.upper(), item.get("score"), item.get("name")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

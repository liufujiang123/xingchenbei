#!/usr/bin/env python3
"""Compact task/run state with evidence freshness and score comparability."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import evidence_fingerprint

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(*args):
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_record(task_dir):
    runs = task_dir / "runs" / "harness"
    if not runs.exists():
        return None, None
    records = sorted(
        p for p in runs.glob("*.json")
        if not p.name.startswith("best-") and p.name not in ("state.json", "context.json")
    )
    if not records:
        return None, None
    path = records[-1]
    return path, load_json(path)


def current_fingerprint(task):
    try:
        config = evidence_fingerprint.resolve_config(task)
        env = evidence_fingerprint.load_env(config)
        return evidence_fingerprint.capture(task, config_path=config, env=env), None
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def stage_views(record, recorded_fp, current):
    stages, freshness = {}, {}
    for stage in (record or {}).get("stages", []):
        name = stage.get("stage")
        if not name:
            continue
        stages[name] = "pass" if stage.get("returncode") == 0 else "fail"
        freshness[name] = evidence_fingerprint.stage_freshness(recorded_fp, current, name)
    return stages, freshness


def overall_freshness(stages, freshness):
    passed = [name for name, status in stages.items() if status == "pass"]
    if not passed:
        return {"status": "unknown", "reasons": ["no_passing_runtime_stage"]}
    states = [freshness.get(name, {}).get("status", "unknown") for name in passed]
    if "stale" in states:
        reasons = []
        for name in passed:
            item = freshness.get(name, {})
            if item.get("status") == "stale":
                reasons.extend("%s:%s" % (name, r) for r in item.get("reasons", []))
        return {"status": "stale", "reasons": reasons}
    if "unknown" in states:
        return {"status": "unknown", "reasons": ["legacy_or_missing_fingerprint"]}
    return {"status": "fresh", "reasons": []}


def enrich_best(item, current, kind):
    if not item:
        return None
    out = dict(item)
    fp = evidence_fingerprint.best_fingerprint(item)
    stage = "platform" if kind == "platform" else "bench"
    out["freshness"] = evidence_fingerprint.stage_freshness(fp, current, stage)
    out["comparability"] = evidence_fingerprint.score_compatibility(fp, current, kind=kind)
    return out


def summarize(task):
    evidence_fingerprint.ROOT = ROOT
    task_dir = ROOT / "tasks" / task
    runs = task_dir / "runs" / "harness"
    latest_path, latest = latest_record(task_dir)
    current, fp_error = current_fingerprint(task)

    latest_view = None
    if latest is not None:
        recorded_fp = evidence_fingerprint.record_fingerprint(latest)
        stages, freshness = stage_views(latest, recorded_fp, current)
        latest_view = {
            "record": str(latest_path.relative_to(ROOT)),
            "mode": latest.get("mode"),
            "name": latest.get("name"),
            "decision": latest.get("decision"),
            "score": latest.get("score"),
            "direction": latest.get("direction"),
            "platform": latest.get("platform"),
            "stages": stages,
            "stage_freshness": freshness,
            "freshness": overall_freshness(stages, freshness),
            "recorded_fingerprint": recorded_fp,
        }

    best_local = load_json(runs / "best-local.json") if runs.exists() else None
    best_platform = load_json(runs / "best-platform.json") if runs.exists() else None

    # Legacy harness stores its best at repo/runs/best.json.
    if best_local is None:
        legacy = load_json(ROOT / "runs" / "best.json")
        if legacy and legacy.get("run"):
            best_local = legacy

    return {
        "task": task,
        "branch": git("branch", "--show-current") or None,
        "head": git("rev-parse", "--short", "HEAD") or None,
        "dirty_paths": len([x for x in git("status", "--short").splitlines() if x.strip()]),
        "fingerprint_error": fp_error,
        "current_fingerprint": current,
        "latest": latest_view,
        "best_local": enrich_best(best_local, current, "bench"),
        "best_platform": enrich_best(best_platform, current, "platform"),
    }


def text_status(item):
    if not item:
        return "unknown"
    status = item.get("status", "unknown")
    reasons = item.get("reasons") or []
    return status if not reasons else "%s(%s)" % (status, ",".join(reasons))


def main():
    parser = argparse.ArgumentParser(description="show compact Harness task state")
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
    if data["fingerprint_error"]:
        print("FINGERPRINT=UNKNOWN error=%s" % data["fingerprint_error"])

    latest = data["latest"]
    if latest:
        print("LATEST mode=%s decision=%s score=%s freshness=%s record=%s" % (
            latest.get("mode"), latest.get("decision"), latest.get("score"),
            text_status(latest.get("freshness")), latest.get("record"),
        ))
        if latest.get("stages"):
            print("STAGES " + " ".join(
                "%s=%s:%s" % (
                    name, status,
                    text_status(latest.get("stage_freshness", {}).get(name)),
                )
                for name, status in latest["stages"].items()
            ))
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
            print("%s score=%s name=%s freshness=%s comparable=%s" % (
                key.upper(), item.get("score"), item.get("name"),
                text_status(item.get("freshness")),
                text_status(item.get("comparability")),
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

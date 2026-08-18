#!/usr/bin/env python3
"""Evidence-driven evaluator for Codex kernel optimization.

Codex edits code. This tool standardizes stage order, captures what actually ran,
and maintains a best-score record without automatically reverting user changes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BEST = RUNS / "best.json"

def shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"

def load_env(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"missing {path}; copy config/agent.env.example to config/agent.env")
    cmd = ["bash", "-lc", f"set -a; source {shq(str(path))}; env -0"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, check=True)
    env: dict[str, str] = {}
    for item in p.stdout.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode()] = v.decode(errors="replace")
    return env

def run_stage(stage: str) -> dict:
    path = ROOT / "scripts" / f"{stage}.sh"
    start = dt.datetime.now(dt.timezone.utc)
    p = subprocess.run(
        ["bash", str(path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    end = dt.datetime.now(dt.timezone.utc)
    return {
        "stage": stage,
        "returncode": p.returncode,
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "output": p.stdout,
    }

def git_meta() -> dict:
    def out(*args: str):
        p = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    return {
        "head": out("rev-parse", "HEAD"),
        "branch": out("branch", "--show-current"),
        "status": out("status", "--short"),
    }

def parse_score(text: str, pattern: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def better(score: float, old: float, direction: str) -> bool:
    if direction == "lower":
        return score < old
    if direction == "higher":
        return score > old
    raise SystemExit("BENCH_DIRECTION must be lower or higher")

def load_best() -> dict | None:
    return json.loads(BEST.read_text()) if BEST.exists() else None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "candidate", "profile"])
    ap.add_argument("--name", default=None, help="candidate label")
    ap.add_argument("--hypothesis", default=None)
    args = ap.parse_args()

    env = load_env(ROOT / "config" / "agent.env")
    RUNS.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "id": stamp,
        "mode": args.mode,
        "name": args.name or args.mode,
        "hypothesis": args.hypothesis,
        "task": env.get("TASK_NAME"),
        "git": git_meta(),
        "stages": [],
    }

    stages = ["guard", "build", "validate"]
    if args.mode in {"baseline", "candidate"}:
        stages.append("bench")
    if args.mode == "profile":
        stages.append("profile")

    for stage in stages:
        result = run_stage(stage)
        record["stages"].append(result)
        sys.stdout.write(result["output"])
        if result["returncode"] != 0:
            record["decision"] = f"reject:{stage}_failed"
            break
        if stage == "bench":
            pattern = env.get("BENCH_SCORE_REGEX", r"SCORE[=:]\s*([0-9]+(?:\.[0-9]+)?)")
            record["score"] = parse_score(result["output"], pattern)
            record["direction"] = env.get("BENCH_DIRECTION", "lower")
            if record["score"] is None:
                record["decision"] = "reject:score_not_parsed"
                break
    else:
        if "score" not in record:
            record["decision"] = "passed"
        else:
            current = load_best()
            score = record["score"]
            direction = record["direction"]
            if current is None:
                record["decision"] = "promote:first_measured_baseline"
            elif better(score, float(current["score"]), direction):
                record["decision"] = "promote:improved"
                record["previous_best"] = current
            else:
                record["decision"] = "reject:not_faster"
                record["current_best"] = current

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["name"])
    out = RUNS / f"{stamp}-{safe_name}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    if record["decision"].startswith("promote:"):
        BEST.write_text(json.dumps({
            "run": str(out.relative_to(ROOT)),
            "name": record["name"],
            "score": record["score"],
            "direction": record["direction"],
            "git": record["git"],
            "hypothesis": record["hypothesis"],
        }, indent=2, ensure_ascii=False) + "\n")

    print(f"\nrecord={out.relative_to(ROOT)}")
    print(f"decision={record['decision']}")
    if "score" in record:
        print(f"score={record['score']} direction={record.get('direction')}")
    return 0 if not record["decision"].startswith("reject:") else 1

if __name__ == "__main__":
    raise SystemExit(main())

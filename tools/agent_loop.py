#!/usr/bin/env python3
"""Task-scoped evidence loop for Ascend competition kernels.

Codex edits code; this tool standardizes gates, records evidence, and keeps
local and CANNJudge best scores separate. Platform submission is never implicit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def shq(value):
    return "'" + value.replace("'", "'\\''") + "'"


def resolve_config(task, explicit=None):
    if explicit:
        path = pathlib.Path(explicit).expanduser()
        return path if path.is_absolute() else ROOT / path
    if task:
        task_path = ROOT / "config" / "tasks" / (task + ".env")
        if task_path.exists():
            return task_path
    legacy = ROOT / "config" / "agent.env"
    if legacy.exists():
        return legacy
    hint = "config/tasks/%s.env" % task if task else "config/agent.env"
    raise SystemExit("missing %s" % hint)


def load_env(path):
    cmd = ["bash", "-lc", "set -a; source %s; env -0" % shq(str(path))]
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, check=True, env=os.environ.copy())
    env = {}
    for item in proc.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode(errors="replace")
    env.update(os.environ)
    return env


def task_paths(task, env):
    raw_task_dir = env.get("TASK_DIR") or ("tasks/%s" % task)
    task_dir = pathlib.Path(raw_task_dir)
    if not task_dir.is_absolute():
        task_dir = ROOT / task_dir
    raw_workspace = env.get("WORKSPACE_DIR") or str(task_dir)
    workspace = pathlib.Path(raw_workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    return task_dir, workspace


def run_stage(stage, command, env, cwd):
    start = dt.datetime.now(dt.timezone.utc)
    print("\n===== %s =====" % stage.upper(), flush=True)
    print("cwd=%s" % cwd, flush=True)
    print("+ %s" % command, flush=True)
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    returncode = proc.wait()
    end = dt.datetime.now(dt.timezone.utc)
    return {
        "stage": stage,
        "command": command,
        "returncode": returncode,
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "output": "".join(lines),
    }


def git_meta():
    def output(*args):
        proc = subprocess.run(["git"] + list(args), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {
        "head": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "status": output("status", "--short"),
    }


def parse_score(text, pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def better(score, old, direction):
    if direction == "lower":
        return score < old
    if direction == "higher":
        return score > old
    raise SystemExit("score direction must be lower or higher")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_best(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def promote_decision(record, best_path, score, direction, prefix):
    current = load_best(best_path)
    if current is None:
        decision = "promote:first_%s_score" % prefix
    elif better(score, float(current["score"]), direction):
        decision = "promote:%s_improved" % prefix
        record["previous_best"] = current
    else:
        decision = "reject:%s_not_improved" % prefix
        record["current_best"] = current
    if decision.startswith("promote:"):
        best_path.write_text(
            json.dumps(
                {
                    "run": record.get("record_path"),
                    "name": record["name"],
                    "score": score,
                    "direction": direction,
                    "git": record["git"],
                    "hypothesis": record.get("hypothesis"),
                    "submission_id": record.get("platform", {}).get("submission_id"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "baseline", "candidate", "profile", "platform"])
    parser.add_argument("--task")
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--hypothesis")
    parser.add_argument("--submit", action="store_true", help="explicitly authorize a CANNJudge submission")
    parser.add_argument("--skip-guard", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    config_path = resolve_config(args.task, args.config)
    env = load_env(config_path)
    task = args.task or env.get("TASK_NAME")
    if not task:
        raise SystemExit("task name is required (--task or TASK_NAME)")
    if args.mode == "platform" and not args.submit:
        raise SystemExit("platform mode performs an external submission; add --submit to authorize it")

    task_dir, workspace = task_paths(task, env)
    runs = task_dir / "runs" / "harness"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or args.mode
    out = runs / (stamp + "-" + safe_name(name) + ".json")
    record = {
        "id": stamp,
        "mode": args.mode,
        "name": name,
        "hypothesis": args.hypothesis,
        "task": task,
        "config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "git": git_meta(),
        "stages": [],
        "record_path": str(out.relative_to(ROOT)),
    }

    run_cwd = ROOT if env.get("RUN_CWD", "workspace") == "repo" else workspace
    commands = {
        "guard": env.get("GUARD_CMD", "bash scripts/guard.sh"),
        "build": env.get("BUILD_CMD", ""),
        "validate": env.get("VALIDATE_CMD", ""),
        "bench": env.get("BENCH_CMD", ""),
        "profile": env.get("PROFILE_CMD", ""),
    }

    stages = []
    if not args.skip_guard:
        stages.append("guard")
    if not args.skip_build:
        stages.append("build")
    if not args.skip_validate:
        stages.append("validate")
    if args.mode in ("baseline", "candidate"):
        stages.append("bench")
    elif args.mode == "profile":
        stages.append("profile")
    elif args.mode == "platform":
        cfg_arg = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path)
        commands["platform"] = "%s tools/cannjudge_eval.py submit --task %s --config %s --yes-submit" % (
            shq(sys.executable), shq(task), shq(cfg_arg)
        )
        stages.append("platform")

    for stage in stages:
        command = commands.get(stage, "").strip()
        if not command:
            record["decision"] = "reject:%s_not_configured" % stage
            break
        result = run_stage(stage, command, env, run_cwd)
        record["stages"].append(result)
        if result["returncode"] != 0:
            record["decision"] = "reject:%s_failed" % stage
            break
        if stage == "bench":
            pattern = env.get("BENCH_SCORE_REGEX", r"SCORE[=:]\s*([0-9]+(?:\.[0-9]+)?)")
            score = parse_score(result["output"], pattern)
            if score is None:
                record["decision"] = "reject:score_not_parsed"
                break
            record["score"] = score
            record["direction"] = env.get("BENCH_DIRECTION", "lower")
        if stage == "platform":
            sid = re.search(r"^CANNJUDGE_SUBMISSION_ID=(\S+)$", result["output"], re.MULTILINE)
            status = re.search(r"^CANNJUDGE_STATUS=(.+)$", result["output"], re.MULTILINE)
            score = re.search(r"^CANNJUDGE_SCORE=([0-9]+(?:\.[0-9]+)?)$", result["output"], re.MULTILINE)
            record["platform"] = {
                "submission_id": sid.group(1) if sid else None,
                "status": status.group(1).strip() if status else "Unknown",
                "score": float(score.group(1)) if score else None,
            }
    else:
        record.setdefault("decision", "passed")

    if "decision" not in record or record["decision"] == "passed":
        if args.mode in ("baseline", "candidate") and "score" in record:
            record["decision"] = promote_decision(
                record, runs / "best-local.json", record["score"], record["direction"], "local"
            )
        elif args.mode == "platform":
            platform = record.get("platform", {})
            status = platform.get("status", "Unknown")
            if status == "Accepted":
                if platform.get("score") is None:
                    record["decision"] = "passed:platform_accepted_no_score"
                else:
                    direction = env.get("PLATFORM_SCORE_DIRECTION", "higher")
                    record["decision"] = promote_decision(
                        record, runs / "best-platform.json", platform["score"], direction, "platform"
                    )
            elif status in ("Running", "Pending", "Queued"):
                record["decision"] = "pending:platform_%s" % status.lower()
            else:
                record["decision"] = "reject:platform_%s" % safe_name(status.lower())
        else:
            record["decision"] = "passed"

    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nrecord=%s" % out.relative_to(ROOT))
    print("decision=%s" % record["decision"])
    if record.get("platform"):
        print("submission_id=%s" % record["platform"].get("submission_id"))
        print("platform_status=%s" % record["platform"].get("status"))
        print("platform_score=%s" % record["platform"].get("score"))
    elif "score" in record:
        print("score=%s direction=%s" % (record["score"], record.get("direction")))

    if record["decision"].startswith("reject:"):
        return 1
    if record["decision"].startswith("pending:"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

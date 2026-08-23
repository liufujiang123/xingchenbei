#!/usr/bin/env python3
"""Task-scoped evidence loop for Ascend competition kernels.

Codex owns engineering decisions. This tool standardizes high-value mechanical
work: task-scoped gates, evidence fingerprints, comparable best-score lineages,
compact design/performance knowledge lookup, and explicitly authorized platform
submission.
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

import ascend_design_analyze
import ascend_perf_analyze
import cannjudge_package
import evidence_fingerprint

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLATFORM_PASS_STATUSES = {"Accepted", "Pass"}


def shq(value):
    return "'" + value.replace("'", "'\\''") + "'"


def resolve_config(task, explicit=None):
    if explicit:
        path = pathlib.Path(explicit).expanduser()
        return path if path.is_absolute() else ROOT / path
    if task:
        path = ROOT / "config" / "tasks" / (task + ".env")
        if path.exists():
            return path
    legacy = ROOT / "config" / "agent.env"
    if legacy.exists():
        return legacy
    hint = "config/tasks/%s.env" % task if task else "config/agent.env"
    raise SystemExit("missing %s" % hint)


def load_env(path):
    proc = subprocess.run(
        ["bash", "-lc", "set -a; source %s; env -0" % shq(str(path))],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
        env=os.environ.copy(),
    )
    env = {}
    for item in proc.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode(errors="replace")
    env.update(os.environ)
    return env


def task_paths(task, env):
    task_dir = pathlib.Path(env.get("TASK_DIR") or ("tasks/%s" % task)).expanduser()
    if not task_dir.is_absolute():
        task_dir = ROOT / task_dir
    workspace = pathlib.Path(env.get("WORKSPACE_DIR") or str(task_dir)).expanduser()
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
        proc = subprocess.run(
            ["git", *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
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
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def archive_best(path, status, kind):
    suffix = status.get("status", "unknown")
    target = path.with_name("%s.%s.json" % (path.stem, suffix))
    index = 1
    while target.exists():
        target = path.with_name("%s.%s.%d.json" % (path.stem, suffix, index))
        index += 1
    path.rename(target)
    print("HARNESS_EVIDENCE_BEST_RESET=%s:%s:%s" % (
        kind, suffix, ",".join(status.get("reasons", [])) or "none"
    ))


def prepare_best_lineage(best_path, current_fp, kind):
    current = load_best(best_path)
    if not current:
        return
    status = evidence_fingerprint.score_compatibility(
        evidence_fingerprint.best_fingerprint(current), current_fp, kind=kind
    )
    if status.get("status") == "compatible":
        print("HARNESS_EVIDENCE_BEST_KEEP=%s" % kind)
    else:
        archive_best(best_path, status, kind)


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
                    "evidence_fingerprint": record.get("evidence_fingerprint"),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return decision


def env_bool(env, key, default=False):
    raw = env.get(key)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def split_list(value):
    if not value:
        return []
    return [item for item in re.split(r"[\s,;:]+", value.strip()) if item]


def configured_source_paths(task_dir, workspace, env, key):
    items = split_list(env.get(key, ""))
    paths = []
    if items:
        for item in items:
            path = pathlib.Path(item).expanduser()
            paths.append(path if path.is_absolute() else ROOT / path)
    else:
        paths = [workspace, task_dir]
    out, seen = [], set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def relative_paths(paths):
    return [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in paths]


def attach_design(record, task_dir, workspace, env, cli_archetypes=None, selected_patterns=None):
    sources = configured_source_paths(task_dir, workspace, env, "DESIGN_SOURCE_DIRS")
    archetypes = split_list(env.get("DESIGN_ARCHETYPES", ""))
    archetypes.extend(cli_archetypes or [])
    limit = max(0, int(env.get("DESIGN_PATTERN_LIMIT", "5").strip()))
    design = ascend_design_analyze.analyze(
        sources,
        archetypes=archetypes,
        limit=limit,
        select_ids=selected_patterns or [],
    )
    design["source_dirs"] = relative_paths(sources)
    record["design"] = design

    print("\n===== ASCEND DESIGN EXPERIENCE CATALOG =====")
    for item in design.get("experience_catalog", []):
        print("%s | %s | %s | %s" % (
            item["id"], item.get("phase"), ",".join(item.get("applies_to", [])), item.get("summary", "")
        ))
    if design.get("machine_suggestions"):
        print("machine_hints=%s" % ",".join(x["id"] for x in design["machine_suggestions"]))
    for item in design.get("selected_decisions", []):
        print("selected_pattern=%s" % item["id"])
        print("decide=%s" % item.get("decide"))
        print("guardrails=%s" % item.get("guardrails"))
        print("validate=%s" % item.get("validate"))
    print("design_rule=%s" % design["rule"])


def profile_output(record):
    return "\n".join(
        stage.get("output", "")
        for stage in record.get("stages", [])
        if stage.get("stage") == "profile"
    )


def attach_diagnosis(record, task_dir, workspace, env, advanced=False):
    configured_class = env.get("PERF_OPERATOR_CLASS", "auto").strip() or "auto"
    if configured_class not in ("auto", "vector", "cube", "mixed_cv"):
        raise ValueError("PERF_OPERATOR_CLASS must be auto/vector/cube/mixed_cv")
    hints = split_list(env.get("PERF_BOTTLENECK_HINTS", ""))
    limit = max(0, int(env.get("PERF_PLAN_LIMIT", "5").strip()))
    include_advanced = advanced or env_bool(env, "PERF_ADVANCED", False)
    sources = configured_source_paths(task_dir, workspace, env, "PERF_SOURCE_DIRS")
    diagnosis = ascend_perf_analyze.analyze(
        sources,
        profile_text=profile_output(record),
        operator_class=configured_class,
        bottleneck_hints=hints,
        include_advanced=include_advanced,
        limit=limit,
    )
    diagnosis["source_dirs"] = relative_paths(sources)
    record["diagnosis"] = diagnosis

    print("\n===== ASCEND DIAGNOSIS =====")
    print("operator_class=%s source=%s confidence=%s" % (
        diagnosis["operator_class"] or "unresolved",
        diagnosis["operator_class_source"],
        diagnosis["confidence"],
    ))
    print("observed_bottlenecks=%s" % (",".join(diagnosis["observed_bottlenecks"]) or "none"))
    print("static_risk_tags=%s" % (",".join(diagnosis["static_risk_tags"]) or "none"))
    for index, candidate in enumerate(diagnosis["candidates"], 1):
        print("candidate_%d=%s matched=%s" % (
            index, candidate["id"], ",".join(candidate["matched_tags"])
        ))
    print("diagnosis_rule=%s" % diagnosis["rule"])


def compact_fp(task, config_path, env, commands):
    evidence_fingerprint.ROOT = ROOT
    return evidence_fingerprint.compact(
        evidence_fingerprint.capture(
            task, config_path=config_path, env=env, commands=commands
        )
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["design", "validate", "baseline", "candidate", "profile", "diagnose", "platform"],
    )
    parser.add_argument("--task")
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--hypothesis")
    parser.add_argument("--archetype", action="append", default=[])
    parser.add_argument(
        "--select-pattern", action="append", default=[],
        help="Codex-selected design experience id; may be repeated (max enforced by analyzer)",
    )
    parser.add_argument("--submit", action="store_true", help="explicitly authorize a CANNJudge submission")
    parser.add_argument("--skip-guard", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--advanced-diagnosis", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
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
        "platform": "",
    }

    if args.mode == "platform":
        cfg_arg = str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path)
        commands["platform"] = "%s tools/cannjudge_eval.py submit --task %s --config %s --yes-submit" % (
            shq(sys.executable), shq(task), shq(cfg_arg)
        )
        try:
            env = dict(env)
            package_sha = cannjudge_package.resolve_package_sha(env)
            env["CANNJUDGE_PACKAGE_SHA256"] = package_sha
            record["platform_package_sha256"] = package_sha
            print("CANNJUDGE_PACKAGE_SHA256=%s" % package_sha)
        except Exception as exc:
            record["decision"] = "reject:platform_package_identity_failed"
            record["platform_package_error"] = "%s: %s" % (type(exc).__name__, exc)
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("decision=%s" % record["decision"])
            print("record=%s" % out.relative_to(ROOT))
            return 1

    initial_fp = compact_fp(task, config_path, env, commands)
    record["evidence_fingerprint"] = initial_fp

    if args.mode in ("baseline", "candidate"):
        prepare_best_lineage(runs / "best-local.json", initial_fp, "bench")
    elif args.mode == "platform":
        prepare_best_lineage(runs / "best-platform.json", initial_fp, "platform")

    stages = []
    if args.mode == "design":
        if not args.skip_guard:
            stages.append("guard")
    else:
        if not args.skip_guard:
            stages.append("guard")
        if not args.skip_build:
            stages.append("build")
        if not args.skip_validate:
            stages.append("validate")
        if args.mode in ("baseline", "candidate"):
            stages.append("bench")
        elif args.mode == "profile" and not args.skip_profile:
            stages.append("profile")
        elif args.mode == "diagnose" and not args.skip_profile and commands["profile"].strip():
            stages.append("profile")
        elif args.mode == "platform":
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
        elif stage == "platform":
            sid = re.search(r"^CANNJUDGE_SUBMISSION_ID=(\S+)$", result["output"], re.MULTILINE)
            status = re.search(r"^CANNJUDGE_STATUS=(.+)$", result["output"], re.MULTILINE)
            score = re.search(r"^CANNJUDGE_SCORE=([0-9]+(?:\.[0-9]+)?)$", result["output"], re.MULTILINE)
            record["platform"] = {
                "submission_id": sid.group(1) if sid else None,
                "status": status.group(1).strip() if status else "Unknown",
                "score": float(score.group(1)) if score else None,
                "package_sha256": record.get("platform_package_sha256"),
            }
    else:
        record.setdefault("decision", "passed")

    final_fp = compact_fp(task, config_path, env, commands)
    record["evidence_fingerprint_end"] = final_fp
    if initial_fp != final_fp and not str(record.get("decision", "")).startswith("reject:"):
        record["decision"] = "reject:evidence_context_changed_during_run"

    if record.get("decision", "passed") == "passed" and args.mode == "design":
        try:
            attach_design(
                record, task_dir, workspace, env,
                cli_archetypes=args.archetype,
                selected_patterns=args.select_pattern,
            )
        except Exception as exc:
            record["design_error"] = "%s: %s" % (type(exc).__name__, exc)
            record["decision"] = "reject:design_failed"

    if record.get("decision", "passed") == "passed" and args.mode in ("profile", "diagnose"):
        try:
            attach_diagnosis(
                record, task_dir, workspace, env,
                advanced=args.advanced_diagnosis,
            )
        except Exception as exc:
            record["diagnosis_error"] = "%s: %s" % (type(exc).__name__, exc)
            record["decision"] = "reject:diagnosis_failed"

    if record.get("decision") == "passed":
        if args.mode in ("baseline", "candidate") and "score" in record:
            record["decision"] = promote_decision(
                record, runs / "best-local.json",
                record["score"], record["direction"], "local",
            )
        elif args.mode == "platform":
            platform = record.get("platform", {})
            status = platform.get("status", "Unknown")
            if status in PLATFORM_PASS_STATUSES:
                if platform.get("score") is None:
                    record["decision"] = "passed:platform_%s_no_score" % status.lower()
                else:
                    direction = env.get("PLATFORM_SCORE_DIRECTION", "higher")
                    record["decision"] = promote_decision(
                        record, runs / "best-platform.json",
                        platform["score"], direction, "platform",
                    )
            elif status in ("Running", "Pending", "Queued"):
                record["decision"] = "pending:platform_%s" % status.lower()
            else:
                record["decision"] = "reject:platform_%s" % safe_name(status.lower())
        elif args.mode == "design":
            record["decision"] = "passed:design"
        elif args.mode == "diagnose":
            record["decision"] = "passed:diagnosis"
        else:
            record["decision"] = "passed"

    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nrecord=%s" % out.relative_to(ROOT))
    print("decision=%s" % record["decision"])
    if record.get("platform"):
        print("submission_id=%s" % record["platform"].get("submission_id"))
        print("platform_status=%s" % record["platform"].get("status"))
        print("platform_score=%s" % record["platform"].get("score"))
        print("platform_package_sha256=%s" % record["platform"].get("package_sha256"))
    elif "score" in record:
        print("score=%s direction=%s" % (record["score"], record.get("direction")))

    if record["decision"].startswith("reject:"):
        return 1
    if record["decision"].startswith("pending:"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bind Harness evidence to the code/execution context that produced it.

This is intentionally mechanical. It does not choose algorithms or optimization
strategies. It only answers two questions:
1) is old build/validate/bench/profile/platform evidence still fresh?
2) are two benchmark/platform scores safe to compare numerically?
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = 1
MARKER = "HARNESS_EVIDENCE_FINGERPRINT_B64="

IGNORE_DIRS = {
    ".git", ".agent-deps", "__pycache__", ".pytest_cache", ".mypy_cache",
    "build", "cmake-build-debug", "cmake-build-release", "output", "dist",
    "runs", "profiles", "msprof_output", "generated",
}
SECRET_RE = re.compile(r"(password|passwd|secret|token|cookie|credential|private[_-]?key)", re.I)
DEFAULT_ENV_KEYS = (
    "ASCEND_ENV_SETUP", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
    "CANN_VERSION", "SOC_VERSION", "ASCEND_SOC_VERSION", "TARGET_SOC",
    "DEVICE_TYPE", "ASCEND_DEVICE_ID", "DEVICE_ID",
    "ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES",
    "TASK_DIR", "WORKSPACE_DIR", "RUN_CWD",
    "BENCH_SCORE_REGEX", "BENCH_DIRECTION",
    "EVIDENCE_CONTEXT_LABEL",
)
PLATFORM_ENV_KEYS = (
    "CANNJUDGE_CONTEST_ID", "CANNJUDGE_PUBLIC_PROBLEM_ID",
    "CANNJUDGE_PROBLEM_ID", "CANNJUDGE_PACKAGE_SHA256",
    "CANNJUDGE_PROJECT_DIR", "PLATFORM_SCORE_DIRECTION",
)
STAGES = ("guard", "build", "validate", "bench", "profile", "platform")


def _json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _split(value):
    if not value:
        return []
    return [x for x in re.split(r"[\s,;:]+", value.strip()) if x]


def _rel(path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _iter_files(paths):
    seen = set()
    for raw in paths:
        path = pathlib.Path(raw)
        if not path.exists():
            continue
        items = [path] if path.is_file() else path.rglob("*")
        for item in items:
            if not item.is_file() or any(part in IGNORE_DIRS for part in item.parts):
                continue
            resolved = item.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield item


def hash_paths(paths):
    h = hashlib.sha256()
    count = 0
    for path in sorted(_iter_files(paths), key=_rel):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        name = _rel(path).encode()
        h.update(len(name).to_bytes(4, "big"))
        h.update(name)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
        count += 1
    h.update(b"FILE_COUNT")
    h.update(str(count).encode())
    return h.hexdigest()


def resolve_config(task, explicit=None):
    if explicit:
        path = pathlib.Path(explicit).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    task_path = ROOT / "config" / "tasks" / (task + ".env")
    if task_path.exists():
        return task_path
    legacy = ROOT / "config" / "agent.env"
    if legacy.exists():
        return legacy
    raise FileNotFoundError("missing config for task %s" % task)


def load_env(path):
    quoted = "'" + str(path).replace("'", "'\\''") + "'"
    proc = subprocess.run(
        ["bash", "-lc", "set -a; source %s; env -0" % quoted],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError("cannot source %s" % path)
    env = {}
    for item in proc.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    # Match task-scoped agent_loop: shell environment may intentionally override config.
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


def commands_from_env(env):
    return {
        "guard": env.get("GUARD_CMD", "bash scripts/guard.sh"),
        "build": env.get("BUILD_CMD", ""),
        "validate": env.get("VALIDATE_CMD", ""),
        "bench": env.get("BENCH_CMD", ""),
        "profile": env.get("PROFILE_CMD", ""),
        "platform": env.get("PLATFORM_CMD", ""),
    }


def _repo_refs(command):
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    out, seen = [], set()
    for token in tokens:
        token = token.strip("(){}[];,")
        if not token or token.startswith("-") or "=" in token:
            continue
        path = pathlib.Path(token).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        try:
            resolved = path.resolve()
            resolved.relative_to(ROOT.resolve())
        except (ValueError, OSError):
            continue
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _stage_hash(stage, command):
    return _json_hash({
        "stage": stage,
        "command": command.strip(),
        "referenced_files": hash_paths(_repo_refs(command)),
    })


def _safe_env(env):
    extra = _split(env.get("EVIDENCE_ENV_KEYS"))
    dynamic = [
        key for key in env
        if re.search(r"(?:^|_)(?:DEVICE_ID|SOC_VERSION|CANN_VERSION)$", key)
    ]
    keys = []
    for key in list(DEFAULT_ENV_KEYS) + dynamic + extra:
        if key not in keys and key in env and not SECRET_RE.search(key):
            keys.append(key)
    return _json_hash({k: env.get(k, "") for k in sorted(keys)}), sorted(keys)


def _platform_identity(env):
    keys = [k for k in PLATFORM_ENV_KEYS if k in env and not SECRET_RE.search(k)]
    return _json_hash({k: env.get(k, "") for k in sorted(keys)}), sorted(keys)


def _paths_from_env(env, key):
    out = []
    for item in _split(env.get(key)):
        path = pathlib.Path(item).expanduser()
        out.append(path if path.is_absolute() else ROOT / path)
    return out


def capture(task, config_path=None, env=None, commands=None):
    config_path = config_path or resolve_config(task)
    env = env or load_env(config_path)
    task_dir, workspace = task_paths(task, env)
    commands = commands or commands_from_env(env)

    subject_paths = _paths_from_env(env, "EVIDENCE_SUBJECT_PATHS") or [workspace]

    case_paths = _paths_from_env(env, "EVIDENCE_CASE_PATHS")
    if not case_paths and (task_dir / "tests").exists():
        case_paths = [task_dir / "tests"]

    context_paths = _paths_from_env(env, "EVIDENCE_CONTEXT_PATHS")
    if not context_paths:
        if (task_dir / "scripts").exists():
            context_paths.append(task_dir / "scripts")
        setup = env.get("ASCEND_ENV_SETUP")
        if setup and pathlib.Path(setup).expanduser().exists():
            context_paths.append(pathlib.Path(setup).expanduser())

    env_hash, env_keys = _safe_env(env)
    platform_hash, platform_keys = _platform_identity(env)
    stage_hashes = {stage: _stage_hash(stage, commands.get(stage, "")) for stage in STAGES}
    subject_hash = hash_paths(subject_paths)
    case_hash = hash_paths(case_paths)
    context_files_hash = hash_paths(context_paths)
    config_hash = hash_paths([config_path])

    build_context = _json_hash({
        "env": env_hash,
        "context_files": context_files_hash, "build": stage_hashes["build"],
    })
    validate_context = _json_hash({
        "build": build_context, "validate": stage_hashes["validate"], "cases": case_hash,
    })
    bench_context = _json_hash({
        "build": build_context, "bench": stage_hashes["bench"], "cases": case_hash,
    })
    profile_context = _json_hash({
        "build": build_context, "profile": stage_hashes["profile"],
    })
    platform_context = _json_hash({
        "platform_identity": platform_hash, "platform": stage_hashes["platform"],
    })
    context_hash = _json_hash({
        "env": env_hash, "config": config_hash, "context_files": context_files_hash,
        "cases": case_hash, "stages": stage_hashes, "platform": platform_hash,
    })

    return {
        "version": VERSION,
        "task": task,
        "subject_hash": subject_hash,
        "context_hash": context_hash,
        "config_hash": config_hash,
        "context_files_hash": context_files_hash,
        "case_hash": case_hash,
        "environment_hash": env_hash,
        "environment_keys": env_keys,
        "platform_identity_hash": platform_hash,
        "platform_identity_keys": platform_keys,
        "stage_hashes": stage_hashes,
        "build_context_hash": build_context,
        "validate_context_hash": validate_context,
        "bench_context_hash": bench_context,
        "profile_context_hash": profile_context,
        "platform_context_hash": platform_context,
        "subject_paths": [_rel(p) for p in subject_paths],
        "case_paths": [_rel(p) for p in case_paths],
        "context_paths": [_rel(p) for p in context_paths],
        "config_path": _rel(config_path),
    }


def compact(fp):
    keys = (
        "version", "task", "subject_hash", "context_hash", "config_hash",
        "context_files_hash", "case_hash", "environment_hash",
        "platform_identity_hash", "stage_hashes", "build_context_hash",
        "validate_context_hash", "bench_context_hash", "profile_context_hash",
        "platform_context_hash",
    )
    return {key: fp.get(key) for key in keys}


def marker(fp):
    raw = json.dumps(compact(fp), sort_keys=True, separators=(",", ":")).encode()
    return MARKER + base64.urlsafe_b64encode(raw).decode()


def parse_marker(text):
    match = re.search(r"^" + re.escape(MARKER) + r"(\S+)$", text or "", re.MULTILINE)
    if not match:
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(match.group(1).encode()).decode())
    except Exception:
        return None


def stage_freshness(recorded, current, stage):
    if not recorded or not current:
        return {"status": "unknown", "reasons": ["fingerprint_missing"]}
    if recorded.get("version") != current.get("version"):
        return {"status": "unknown", "reasons": ["fingerprint_version_mismatch"]}

    reasons = []
    if recorded.get("subject_hash") != current.get("subject_hash"):
        reasons.append("subject_changed")
    key = {
        "guard": "context_hash",
        "build": "build_context_hash",
        "validate": "validate_context_hash",
        "bench": "bench_context_hash",
        "profile": "profile_context_hash",
        "platform": "platform_context_hash",
    }.get(stage, "context_hash")
    if recorded.get(key) != current.get(key):
        if stage != "platform" and recorded.get("environment_hash") != current.get("environment_hash"):
            reasons.append("environment_changed")
        if stage != "platform" and recorded.get("config_hash") != current.get("config_hash"):
            reasons.append("config_changed")
        if stage != "platform" and recorded.get("context_files_hash") != current.get("context_files_hash"):
            reasons.append("gate_files_changed")
        if stage in ("validate", "bench") and recorded.get("case_hash") != current.get("case_hash"):
            reasons.append("case_set_changed")
        if stage == "platform" and recorded.get("platform_identity_hash") != current.get("platform_identity_hash"):
            reasons.append("platform_identity_changed")
        if recorded.get("stage_hashes", {}).get(stage) != current.get("stage_hashes", {}).get(stage):
            reasons.append("%s_command_changed" % stage)
        if not reasons or reasons == ["subject_changed"]:
            reasons.append("execution_context_changed")
    return {"status": "fresh" if not reasons else "stale", "reasons": reasons}


def score_compatibility(a, b, kind="bench"):
    if not a or not b:
        return {"status": "unknown", "reasons": ["fingerprint_missing"]}
    if a.get("version") != b.get("version"):
        return {"status": "unknown", "reasons": ["fingerprint_version_mismatch"]}
    key = "platform_context_hash" if kind == "platform" else "bench_context_hash"
    if a.get(key) == b.get(key):
        return {"status": "compatible", "reasons": []}

    reasons = []
    if kind == "platform":
        if a.get("platform_identity_hash") != b.get("platform_identity_hash"):
            reasons.append("platform_identity_changed")
        if a.get("stage_hashes", {}).get("platform") != b.get("stage_hashes", {}).get("platform"):
            reasons.append("platform_command_changed")
    else:
        if a.get("environment_hash") != b.get("environment_hash"):
            reasons.append("environment_changed")
        if a.get("config_hash") != b.get("config_hash"):
            reasons.append("config_changed")
        if a.get("context_files_hash") != b.get("context_files_hash"):
            reasons.append("gate_files_changed")
        if a.get("case_hash") != b.get("case_hash"):
            reasons.append("case_set_changed")
        if a.get("stage_hashes", {}).get("build") != b.get("stage_hashes", {}).get("build"):
            reasons.append("build_command_changed")
        if a.get("stage_hashes", {}).get("bench") != b.get("stage_hashes", {}).get("bench"):
            reasons.append("bench_command_changed")
    return {"status": "incompatible", "reasons": reasons or ["execution_context_changed"]}


def record_fingerprint(record):
    if not record:
        return None
    if record.get("evidence_fingerprint"):
        return record["evidence_fingerprint"]
    for stage in record.get("stages", []):
        if stage.get("stage") == "guard":
            fp = parse_marker(stage.get("output", ""))
            if fp:
                return fp
    return parse_marker(record.get("output", ""))


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def best_fingerprint(best):
    if not best:
        return None
    if best.get("evidence_fingerprint"):
        return best["evidence_fingerprint"]
    run = best.get("run")
    if not run:
        return None
    path = pathlib.Path(run)
    if not path.is_absolute():
        path = ROOT / path
    return record_fingerprint(load_json(path))


def _archive_best(path, status, kind):
    stem = path.stem
    suffix = status.get("status", "unknown")
    target = path.with_name("%s.%s.json" % (stem, suffix))
    index = 1
    while target.exists():
        target = path.with_name("%s.%s.%d.json" % (stem, suffix, index))
        index += 1
    path.rename(target)
    print("HARNESS_EVIDENCE_BEST_RESET=%s:%s:%s" % (
        kind, status.get("status"), ",".join(status.get("reasons", [])) or "none"
    ))


def prepare_best(task, current):
    task_dir, _ = task_paths(task, load_env(resolve_config(task)))
    candidates = [
        (task_dir / "runs" / "harness" / "best-local.json", "bench"),
        (task_dir / "runs" / "harness" / "best-platform.json", "platform"),
    ]
    legacy = ROOT / "runs" / "best.json"
    if legacy.exists():
        candidates.append((legacy, "bench"))

    for path, kind in candidates:
        if not path.exists():
            continue
        best = load_json(path)
        status = score_compatibility(best_fingerprint(best), current, kind=kind)
        if status.get("status") == "compatible":
            print("HARNESS_EVIDENCE_BEST_KEEP=%s" % kind)
        else:
            _archive_best(path, status, kind)


def main():
    parser = argparse.ArgumentParser(description="capture/compare Harness evidence context")
    parser.add_argument("--task", required=True)
    parser.add_argument("--config")
    parser.add_argument("--emit-marker", action="store_true")
    parser.add_argument("--prepare-best", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    config = resolve_config(args.task, args.config)
    env = load_env(config)
    current = capture(args.task, config_path=config, env=env)

    if args.prepare_best:
        prepare_best(args.task, current)
    if args.emit_marker:
        print(marker(current))
    elif args.as_json:
        print(json.dumps(current, ensure_ascii=False, indent=2))
    else:
        print("TASK=%s" % current["task"])
        print("subject_hash=%s" % current["subject_hash"])
        print("context_hash=%s" % current["context_hash"])
        print("bench_context_hash=%s" % current["bench_context_hash"])
        print("case_hash=%s" % current["case_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

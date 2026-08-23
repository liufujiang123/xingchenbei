#!/usr/bin/env python3
"""Safe CANNJudge adapter used by the repository harness.

It reuses the official cann-learning-hub cannjudge-submit client. Plaintext
password automation is deliberately unsupported: use RSA ciphertext login.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL_STATUSES = {
    "Accepted",
    "Pass",
    "Wrong Answer",
    "Compile Error",
    "Runtime Error",
    "Time Limit Exceeded",
    "Memory Limit Exceeded",
    "System Error",
    "Cancelled",
}


def shq(value):
    return "'" + value.replace("'", "'\\''") + "'"


def resolve_config(task, explicit=None):
    if explicit:
        path = pathlib.Path(explicit).expanduser()
        return path if path.is_absolute() else ROOT / path
    task_path = ROOT / "config" / "tasks" / (task + ".env")
    if task_path.exists():
        return task_path
    legacy = ROOT / "config" / "agent.env"
    if legacy.exists():
        return legacy
    raise SystemExit("missing task config: config/tasks/%s.env" % task)


def load_env(path):
    cmd = ["bash", "-lc", "set -a; source %s; env -0" % shq(str(path))]
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, check=True, env=os.environ.copy())
    env = {}
    for item in proc.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode(errors="replace")
    # Explicit caller variables (credentials/device overrides) win over tracked defaults.
    env.update(os.environ)
    return env


def as_path(raw):
    path = pathlib.Path(os.path.expanduser(os.path.expandvars(raw)))
    return path if path.is_absolute() else ROOT / path


def discover_skill(env):
    candidates = []
    if env.get("CANNJUDGE_SKILL_DIR"):
        candidates.append(as_path(env["CANNJUDGE_SKILL_DIR"]))
    candidates += [
        ROOT / ".agents" / "skills" / "cannjudge-submit",
        ROOT / ".agent-deps" / "cann-learning-hub" / "skills" / "cannjudge-submit",
    ]
    for candidate in candidates:
        if (candidate / "cannjudge_cli.py").is_file():
            return candidate.resolve()
    raise SystemExit("official cannjudge-submit skill not found; run: bash scripts/bootstrap_skills.sh")


def load_official_module(skill_dir):
    cli = skill_dir / "cannjudge_cli.py"
    spec = importlib.util.spec_from_file_location("xingchenbei_cannjudge_cli", str(cli))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import official CANNJudge client: %s" % cli)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_sources(project_dir):
    code = project_dir / "code" if (project_dir / "code").is_dir() else project_dir
    tiling = sorted((code / "op_kernel").glob("*_tiling.h"))
    tiling_key = sorted((code / "op_kernel").glob("tiling_key_*.h"))
    host = sorted((code / "op_host").glob("*.cpp"))
    kernel = sorted(p for p in (code / "op_kernel").glob("*.cpp") if "_tiling" not in p.name)
    if len(tiling) != 1 or len(host) != 1 or len(kernel) != 1 or len(tiling_key) > 1:
        raise SystemExit(
            "ambiguous submission source: tiling=%d tiling_key=%d host=%d kernel=%d"
            % (len(tiling), len(tiling_key), len(host), len(kernel))
        )
    return {
        "tiling_h": tiling[0],
        "tiling_key_h": tiling_key[0] if tiling_key else None,
        "host_cpp": host[0],
        "kernel_cpp": kernel[0],
    }


def sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_path(env, skill_dir):
    return as_path(env["CANNJUDGE_PRIVATE_KEY"]).resolve() if env.get("CANNJUDGE_PRIVATE_KEY") else (skill_dir / "private.pem").resolve()


def ensure_key(path):
    if not path.is_file():
        raise SystemExit("private key not found: %s" % path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SystemExit("private key permissions are too broad (%04o); run chmod 600 %s" % (mode, path))


def authenticate(module, skill_dir, env, interactive=True):
    if env.get("CANNJUDGE_PASSWORD"):
        raise SystemExit("CANNJUDGE_PASSWORD is blocked; use RSA ciphertext login")
    email = env.get("CANNJUDGE_EMAIL", "").strip()
    ciphertext = env.get("CANNJUDGE_CIPHERTEXT", "").strip()
    ciphertext_file = env.get("CANNJUDGE_CIPHERTEXT_FILE", "").strip()
    if not ciphertext and ciphertext_file:
        path = as_path(ciphertext_file)
        if not path.is_file():
            raise SystemExit("ciphertext file not found: %s" % path)
        ciphertext = path.read_text(encoding="utf-8").strip()
    if interactive and sys.stdin.isatty():
        if not email:
            email = input("CANNJudge email: ").strip()
        if not ciphertext:
            ciphertext = getpass.getpass("CANNJudge RSA ciphertext: ").strip()
    if not email:
        raise SystemExit("missing CANNJUDGE_EMAIL")
    if not ciphertext:
        raise SystemExit("missing CANNJUDGE_CIPHERTEXT_FILE or CANNJUDGE_CIPHERTEXT")
    private_key = key_path(env, skill_dir)
    ensure_key(private_key)
    if not getattr(module, "HAS_CRYPTO", False):
        raise SystemExit("pycryptodome is required for RSA ciphertext login")
    client = module.CANNJudgeClient()
    client.login_with_ciphertext(email, ciphertext, str(private_key))
    return client


def status_of(value):
    if isinstance(value, dict):
        if isinstance(value.get("status"), str):
            return value["status"]
        if isinstance(value.get("data"), dict):
            return status_of(value["data"])
    return "Unknown"


def number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    return None


def score_of(value):
    if not isinstance(value, dict):
        return None
    for key in ("score", "totalScore", "total_score", "points"):
        if key in value:
            parsed = number(value[key])
            if parsed is not None:
                return parsed
    return score_of(value.get("data")) if isinstance(value.get("data"), dict) else None


def ranking_items(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "result", "items", "list"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    return []


def ranking_match(rankings, submission_id, user_id):
    items = ranking_items(rankings)
    for item in items:
        if any(str(item.get(key, "")) == submission_id for key in ("submissionId", "submission_id", "_id", "id")):
            return item
    if user_id:
        for item in items:
            candidates = [item.get("userId"), item.get("user_id")]
            if isinstance(item.get("user"), dict):
                candidates += [item["user"].get("_id"), item["user"].get("id")]
            if any(str(candidate) == str(user_id) for candidate in candidates if candidate is not None):
                return item
    return None


def wait_result(client, submission_id, max_wait, interval):
    deadline = time.monotonic() + max_wait
    last = {}
    while True:
        last = client.get_submission(submission_id)
        status = status_of(last)
        print("CANNJudge status: %s" % status, file=sys.stderr, flush=True)
        if status in TERMINAL_STATUSES or time.monotonic() >= deadline:
            return last
        time.sleep(interval)


def git_head():
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout.strip() if proc.returncode == 0 else None


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def problem_identity_mismatches(problem, env):
    if not isinstance(problem, dict):
        return ["platform problem response is not an object"]
    expected = {
        "internal_id": env.get("CANNJUDGE_PROBLEM_ID", "").strip(),
        "public_id": env.get("CANNJUDGE_PUBLIC_PROBLEM_ID", "").strip(),
        "contest_id": env.get("CANNJUDGE_CONTEST_ID", "").strip(),
    }
    observed = {
        "internal_id": str(problem.get("_id", "")).strip(),
        "public_id": str(problem.get("ID", "")).strip(),
        "contest_id": str(problem.get("contest_id", "")).strip(),
    }
    return [
        "%s expected=%s observed=%s" % (key, expected[key], observed[key])
        for key in expected
        if expected[key] and expected[key] != observed[key]
    ]


def print_problem_identity(problem):
    print("platform.internal_id=%s" % problem.get("_id", "<missing>"))
    print("platform.public_id=%s" % problem.get("ID", "<missing>"))
    print("platform.contest_id=%s" % problem.get("contest_id", "<missing>"))


def command_doctor(args, env, config_path):
    skill = discover_skill(env)
    module = load_official_module(skill)
    problem_id = env.get("CANNJUDGE_PROBLEM_ID", "").strip()
    project_raw = env.get("CANNJUDGE_PROJECT_DIR", "").strip()
    if not problem_id or not project_raw:
        raise SystemExit("CANNJUDGE_PROBLEM_ID and CANNJUDGE_PROJECT_DIR are required")
    project = as_path(project_raw)
    sources = project_sources(project)
    private_key = key_path(env, skill)
    print("task=%s" % args.task)
    print("config=%s" % relative(config_path))
    print("skill_dir=%s" % skill)
    print("problem_id=%s" % problem_id)
    print("project_dir=%s" % project)
    for name, path in sources.items():
        print("source.%s=%s" % (name, path if path else "<empty>"))
    print("pycryptodome=%s" % ("ok" if getattr(module, "HAS_CRYPTO", False) else "missing"))
    print("private_key=%s" % private_key)
    print("private_key_ready=%s" % ("yes" if private_key.is_file() else "no"))
    print("email_ready=%s" % ("yes" if env.get("CANNJUDGE_EMAIL") else "no"))
    print("ciphertext_ready=%s" % ("yes" if env.get("CANNJUDGE_CIPHERTEXT") or env.get("CANNJUDGE_CIPHERTEXT_FILE") else "no"))
    if env.get("CANNJUDGE_PASSWORD"):
        print("plaintext_password_env=BLOCKED")
        return 2
    if private_key.is_file():
        ensure_key(private_key)
    if not getattr(module, "HAS_CRYPTO", False):
        return 2
    try:
        problem = module.CANNJudgeClient().get_problem(problem_id)
    except Exception as exc:
        print("platform_identity_check=FAIL (%s: %s)" % (type(exc).__name__, exc))
        return 2
    print_problem_identity(problem)
    mismatches = problem_identity_mismatches(problem, env)
    if mismatches:
        for mismatch in mismatches:
            print("platform_identity_mismatch=%s" % mismatch)
        print("platform_identity_check=FAIL")
        return 2
    print("platform_identity_check=PASS")
    print("CANNJUDGE_HARNESS_DOCTOR=PASS")
    return 0


def command_submit(args, env):
    if not args.yes_submit:
        raise SystemExit("external submission requires explicit --yes-submit")
    problem_id = env.get("CANNJUDGE_PROBLEM_ID", "").strip()
    project_raw = env.get("CANNJUDGE_PROJECT_DIR", "").strip()
    if not problem_id or not project_raw:
        raise SystemExit("CANNJUDGE_PROBLEM_ID and CANNJUDGE_PROJECT_DIR are required")
    project = as_path(project_raw)
    sources = project_sources(project)
    skill = discover_skill(env)
    module = load_official_module(skill)
    client = authenticate(module, skill, env, interactive=True)
    print("CANNJUDGE_LOGIN=ok", flush=True)

    problem = client.get_problem(problem_id)
    mismatches = problem_identity_mismatches(problem, env)
    if mismatches:
        raise SystemExit("platform problem identity mismatch: %s" % "; ".join(mismatches))
    canonical_id = str(problem.get("_id", problem_id)) if isinstance(problem, dict) else problem_id
    print("CANNJUDGE_PROBLEM_ID=%s" % canonical_id, flush=True)
    if isinstance(problem, dict) and problem.get("name"):
        print("CANNJUDGE_PROBLEM_NAME=%s" % problem["name"], flush=True)

    submission_id = client.submit(
        canonical_id,
        sources["tiling_h"].read_text(encoding="utf-8"),
        sources["tiling_key_h"].read_text(encoding="utf-8") if sources["tiling_key_h"] else "",
        sources["host_cpp"].read_text(encoding="utf-8"),
        sources["kernel_cpp"].read_text(encoding="utf-8"),
    )
    print("CANNJUDGE_SUBMISSION_ID=%s" % submission_id, flush=True)
    max_wait = args.max_wait if args.max_wait is not None else int(env.get("CANNJUDGE_MAX_WAIT", "600"))
    interval = args.interval if args.interval is not None else int(env.get("CANNJUDGE_POLL_INTERVAL", "3"))
    result = wait_result(client, submission_id, max_wait, interval)
    status = status_of(result)
    score = score_of(result)
    matched = None
    try:
        matched = ranking_match(client.get_rankings(canonical_id), submission_id, getattr(client, "user_id", None))
        if score is None and matched is not None:
            score = score_of(matched)
    except Exception as exc:
        print("ranking lookup warning: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)

    out_dir = ROOT / "tasks" / args.task / "runs" / "cannjudge"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "task": args.task,
        "problem_id": canonical_id,
        "public_problem_id": env.get("CANNJUDGE_PUBLIC_PROBLEM_ID"),
        "submission_id": submission_id,
        "status": status,
        "score": score,
        "git_head": git_head(),
        "project_dir": relative(project),
        "sources": {
            name: {"path": relative(path) if path else None, "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "result": result,
        "ranking_match": matched,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = out_dir / (stamp + "-" + submission_id + ".json")
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CANNJUDGE_STATUS=%s" % status, flush=True)
    print("CANNJUDGE_SCORE=%s" % (score if score is not None else "NA"), flush=True)
    print("CANNJUDGE_EVIDENCE=%s" % relative(out), flush=True)
    return 0


def command_query(args, env):
    if not args.submission_id:
        raise SystemExit("query requires --submission-id")
    skill = discover_skill(env)
    module = load_official_module(skill)
    client = authenticate(module, skill, env, interactive=True)
    result = client.get_submission(args.submission_id)
    print("CANNJUDGE_SUBMISSION_ID=%s" % args.submission_id)
    print("CANNJUDGE_STATUS=%s" % status_of(result))
    score = score_of(result)
    print("CANNJUDGE_SCORE=%s" % (score if score is not None else "NA"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_rank(args, env):
    problem_id = env.get("CANNJUDGE_PROBLEM_ID", "").strip()
    if not problem_id:
        raise SystemExit("missing CANNJUDGE_PROBLEM_ID")
    skill = discover_skill(env)
    module = load_official_module(skill)
    client = authenticate(module, skill, env, interactive=True)
    items = ranking_items(client.get_rankings(problem_id))
    for index, item in enumerate(items[: args.limit], 1):
        user = item.get("user_id", item.get("userId", "unknown"))
        print("%3d status=%s score=%s user=%s" % (index, item.get("status"), score_of(item), user))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Repository-safe CANNJudge adapter")
    parser.add_argument("command", choices=["doctor", "submit", "query", "rank"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--config")
    parser.add_argument("--yes-submit", action="store_true")
    parser.add_argument("--submission-id")
    parser.add_argument("--max-wait", type=int)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    config_path = resolve_config(args.task, args.config)
    env = load_env(config_path)
    if args.command == "doctor":
        return command_doctor(args, env, config_path)
    if args.command == "submit":
        return command_submit(args, env)
    if args.command == "query":
        return command_query(args, env)
    return command_rank(args, env)


if __name__ == "__main__":
    raise SystemExit(main())

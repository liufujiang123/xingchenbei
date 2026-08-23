#!/usr/bin/env python3
"""Session-scoped context reuse for Codex.

The model context is the primary cache. This tool only remembers which file
version was already presented in the current coding session and, on change,
shows a diff instead of encouraging a full reread.

State lives under .git/xingchen-context and is never committed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pathlib
import re
import secrets
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".git" / "xingchen-context"
MAX_DIFF_LINES = 240
DEFAULT_RELATIVE = (
    "AGENTS.md",
    ".agents/skills/xingchen-kernel-optimizer/SKILL.md",
)


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def branch_key() -> str:
    branch = git("branch", "--show-current") or "detached"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", branch)


def task_dir(task: str) -> pathlib.Path:
    return ROOT / "tasks" / task


def session_root(task: str) -> pathlib.Path:
    return CACHE_ROOT / branch_key() / re.sub(r"[^A-Za-z0-9_.-]+", "_", task)


def active_file(task: str) -> pathlib.Path:
    return session_root(task) / "active-session"


def new_session(task: str) -> str:
    sid = "%s-%s" % (time.strftime("%Y%m%dT%H%M%S"), secrets.token_hex(3))
    base = session_root(task)
    base.mkdir(parents=True, exist_ok=True)
    active_file(task).write_text(sid + "\n", encoding="utf-8")
    (base / sid / "snapshots").mkdir(parents=True, exist_ok=True)
    save_ledger(task, sid, {"version": 1, "task": task, "branch": branch_key(), "files": {}})
    return sid


def current_session(task: str, create: bool = True) -> str:
    path = active_file(task)
    if path.exists():
        sid = path.read_text(encoding="utf-8").strip()
        if sid:
            return sid
    if not create:
        raise SystemExit("no active context session; run bootstrap --new-session")
    return new_session(task)


def ledger_path(task: str, sid: str) -> pathlib.Path:
    return session_root(task) / sid / "ledger.json"


def load_ledger(task: str, sid: str) -> dict:
    path = ledger_path(task, sid)
    if not path.exists():
        return {"version": 1, "task": task, "branch": branch_key(), "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(task: str, sid: str, data: dict) -> None:
    path = ledger_path(task, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.absolute().relative_to(ROOT.absolute()))
    except ValueError:
        return str(path.absolute())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_path(task: str, sid: str, key: str) -> pathlib.Path:
    safe = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return session_root(task) / sid / "snapshots" / (safe + ".txt")


def current_info(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"exists": False, "hash": None}
    raw = path.read_bytes()
    return {"exists": True, "hash": digest(raw), "size": len(raw)}


def default_paths(task: str) -> list[pathlib.Path]:
    items = [resolve_path(x) for x in DEFAULT_RELATIVE]
    for name in ("TASK.md", "design.md"):
        path = task_dir(task) / name
        if path.exists():
            items.append(path)
    return items


def diff_text(old: str, new: str, label: str) -> tuple[str, bool]:
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=label + " (previously presented)",
            tofile=label + " (current)",
            lineterm="",
        )
    )
    if len(lines) <= MAX_DIFF_LINES:
        return "\n".join(lines), False
    return "\n".join(lines[:MAX_DIFF_LINES]), True


def present(task: str, sid: str, path: pathlib.Path, force: bool = False) -> str:
    ledger = load_ledger(task, sid)
    key = rel(path)
    info = current_info(path)
    previous = ledger.setdefault("files", {}).get(key)

    if not info["exists"]:
        return "MISSING %s" % key
    if previous and previous.get("hash") == info["hash"] and not force:
        return "REUSE_CONTEXT %s hash=%s" % (key, info["hash"][:12])

    text = path.read_text(encoding="utf-8", errors="replace")
    snap = snapshot_path(task, sid, key)
    header = []
    if force or not previous or not snap.exists():
        header.append("READ_ONCE %s hash=%s" % (key, info["hash"][:12]))
        body = text
    else:
        old = snap.read_text(encoding="utf-8", errors="replace")
        body, truncated = diff_text(old, text, key)
        header.append("CHANGED %s hash=%s; consume diff first" % (key, info["hash"][:12]))
        if truncated:
            header.append(
                "DIFF_TRUNCATED after %d lines; read only the section needed, or use --force for exact recovery"
                % MAX_DIFF_LINES
            )
        if not body:
            body = "(content hash changed but no textual diff was produced)"

    snap.write_text(text, encoding="utf-8")
    ledger["files"][key] = {
        "hash": info["hash"],
        "size": info["size"],
        "presented_at": int(time.time()),
        "git_head": git("rev-parse", "HEAD"),
    }
    save_ledger(task, sid, ledger)
    return "\n".join(header) + "\n---\n" + body + "\n---"


def check(task: str, sid: str, paths: list[pathlib.Path]) -> list[dict]:
    ledger = load_ledger(task, sid)
    result = []
    for path in paths:
        key = rel(path)
        info = current_info(path)
        previous = ledger.get("files", {}).get(key)
        if not info["exists"]:
            status, action = "missing", "resolve_if_needed"
        elif not previous:
            status, action = "not_presented", "read_once"
        elif previous.get("hash") == info["hash"]:
            status, action = "unchanged", "reuse_context"
        else:
            status, action = "changed", "use_context_state_diff"
        result.append({"path": key, "status": status, "action": action, "hash": info.get("hash")})
    return result


def cmd_bootstrap(args) -> int:
    sid = new_session(args.task) if args.new_session else current_session(args.task)
    print("CONTEXT_SESSION=%s task=%s branch=%s" % (sid, args.task, branch_key()))
    paths = default_paths(args.task)
    paths.extend(resolve_path(x) for x in args.include)
    seen = set()
    for path in paths:
        key = rel(path)
        if key in seen:
            continue
        seen.add(key)
        print()
        print(present(args.task, sid, path, force=args.force))
    print()
    print("CONTEXT_RULE=unchanged:reuse; changed:diff-first; full-reread:last-resort")
    return 0


def cmd_use(args) -> int:
    sid = current_session(args.task)
    for raw in args.paths:
        print(present(args.task, sid, resolve_path(raw), force=args.force))
    return 0


def cmd_check(args) -> int:
    sid = current_session(args.task)
    paths = default_paths(args.task)
    paths.extend(resolve_path(x) for x in args.include)
    rows = check(args.task, sid, paths)
    if args.as_json:
        print(json.dumps({"session": sid, "task": args.task, "files": rows}, ensure_ascii=False, indent=2))
    else:
        print("CONTEXT_SESSION=%s" % sid)
        for row in rows:
            print("%-12s %-20s %s" % (row["status"].upper(), row["action"], row["path"]))
        reread = [x["path"] for x in rows if x["status"] in ("changed", "not_presented")]
        print("ACTION_REQUIRED=%s" % (",".join(reread) if reread else "NONE"))
    return 0


def cmd_reset(args) -> int:
    sid = new_session(args.task)
    print("CONTEXT_SESSION_RESET=%s" % sid)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="session-scoped Codex context reuse")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap")
    boot.add_argument("--task", required=True)
    boot.add_argument("--new-session", action="store_true")
    boot.add_argument("--include", action="append", default=[])
    boot.add_argument("--force", action="store_true")
    boot.set_defaults(func=cmd_bootstrap)

    use = sub.add_parser("use")
    use.add_argument("--task", required=True)
    use.add_argument("--force", action="store_true")
    use.add_argument("paths", nargs="+")
    use.set_defaults(func=cmd_use)

    chk = sub.add_parser("check")
    chk.add_argument("--task", required=True)
    chk.add_argument("--include", action="append", default=[])
    chk.add_argument("--json", action="store_true", dest="as_json")
    chk.set_defaults(func=cmd_check)

    reset = sub.add_parser("reset")
    reset.add_argument("--task", required=True)
    reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

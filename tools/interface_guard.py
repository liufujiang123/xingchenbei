#!/usr/bin/env python3
"""Snapshot/check immutable platform-visible files without guessing semantics.

The guard is opt-in: copy config/protected_paths.example to config/protected_paths.txt
after importing the official competition template, list only truly immutable files,
then run `python3 tools/interface_guard.py snapshot`.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "protected_paths.txt"
SNAPSHOT = ROOT / "config" / "protected_paths.sha256.json"

def load_paths() -> list[pathlib.Path]:
    if not CFG.exists():
        raise SystemExit(f"missing {CFG}; guard is not configured")
    result = []
    for raw in CFG.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        p = (ROOT / s).resolve()
        try:
            p.relative_to(ROOT.resolve())
        except ValueError:
            raise SystemExit(f"path escapes repository: {s}")
        if not p.is_file():
            raise SystemExit(f"protected file missing: {s}")
        result.append(p)
    if not result:
        raise SystemExit(f"no protected paths configured in {CFG}")
    return result

def digest(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))

def snapshot() -> int:
    data = {rel(p): digest(p) for p in load_paths()}
    SNAPSHOT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)} with {len(data)} files")
    return 0

def check() -> int:
    if not SNAPSHOT.exists():
        raise SystemExit(f"missing {SNAPSHOT}; run snapshot first")
    expected = json.loads(SNAPSHOT.read_text())
    failures = []
    for name, old in expected.items():
        p = ROOT / name
        if not p.is_file():
            failures.append(f"missing: {name}")
            continue
        new = digest(p)
        if new != old:
            failures.append(f"changed: {name}")
    if failures:
        print("interface guard failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"interface guard passed ({len(expected)} files)")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["snapshot", "check"])
    args = ap.parse_args()
    return snapshot() if args.action == "snapshot" else check()

if __name__ == "__main__":
    raise SystemExit(main())

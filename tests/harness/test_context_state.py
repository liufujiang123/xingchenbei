#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile


def load_module(path):
    spec = importlib.util.spec_from_file_location("context_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    src = pathlib.Path(__file__).resolve().parents[2] / "tools" / "context_state.py"
    mod = load_module(src)

    root = pathlib.Path(tempfile.mkdtemp(prefix="context-state-test-"))
    (root / ".git").mkdir()
    (root / "tasks" / "demo").mkdir(parents=True)
    (root / ".agents" / "skills" / "xingchen-kernel-optimizer").mkdir(parents=True)
    (root / "AGENTS.md").write_text("policy v1\n", encoding="utf-8")
    (root / ".agents" / "skills" / "xingchen-kernel-optimizer" / "SKILL.md").write_text("skill v1\n", encoding="utf-8")
    (root / "tasks" / "demo" / "TASK.md").write_text("task v1\n", encoding="utf-8")
    (root / "tasks" / "demo" / "design.md").write_text("design v1\n", encoding="utf-8")

    mod.ROOT = root
    mod.CACHE_ROOT = root / ".git" / "xingchen-context"
    mod.git = lambda *args: "test-branch" if args[:2] == ("branch", "--show-current") else "deadbeef"

    sid = mod.new_session("demo")
    first = mod.present("demo", sid, root / "AGENTS.md")
    assert first.startswith("READ_ONCE")
    again = mod.present("demo", sid, root / "AGENTS.md")
    assert again.startswith("REUSE_CONTEXT")

    (root / "AGENTS.md").write_text("policy v2\n", encoding="utf-8")
    changed = mod.present("demo", sid, root / "AGENTS.md")
    assert changed.startswith("CHANGED")
    assert "-policy v1" in changed and "+policy v2" in changed

    rows = mod.check("demo", sid, [root / "AGENTS.md", root / "tasks" / "demo" / "TASK.md"])
    by_path = {x["path"]: x for x in rows}
    assert by_path["AGENTS.md"]["status"] == "unchanged"
    assert by_path["tasks/demo/TASK.md"]["status"] == "not_presented"

    sid2 = mod.new_session("demo")
    assert sid2 != sid
    rows2 = mod.check("demo", sid2, [root / "AGENTS.md"])
    assert rows2[0]["status"] == "not_presented"

    print("PASS context_state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

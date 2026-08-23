# Codex workflow

The authoritative workflow is intentionally short.

1. Follow root/nested `AGENTS.md` and the task's authoritative competition contract/template.
2. At the start of a new task conversation, run `context_state.py bootstrap --new-session` once; reuse unchanged context instead of rereading stable files.
3. Use `task_state.py` before citing old build/correctness/performance evidence.
4. Let Codex handle ordinary implementation/debugging. Use `xingchen-kernel-optimizer` and the design/performance catalogs when Ascend-specific knowledge or bottleneck reasoning is useful.
5. Run target build -> correctness -> same-case benchmark. Profile only for a concrete unanswered bottleneck question.
6. Never claim local proxy results as target-platform proof.
7. Never submit externally unless the user explicitly authorizes it; `agent_loop.py platform` requires `--submit`.

Typical commands:

```bash
python3 tools/context_state.py bootstrap --task <task> --new-session
python3 tools/task_state.py --task <task>
python3 tools/agent_loop.py validate --task <task>
python3 tools/agent_loop.py baseline --task <task> --name baseline
python3 tools/agent_loop.py diagnose --task <task>
python3 tools/agent_loop.py candidate --task <task> --name <candidate> --hypothesis '<one mechanism>'
```

Use `python3 tools/agent_loop.py --help` and the relevant Skill when more detail is actually needed; do not preload every document or experience body.

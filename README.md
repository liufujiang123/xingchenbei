# Xingchenbei Ascend Kernel Agent Harness

A compact Codex-oriented harness for developing and optimizing Ascend C competition operators.

The repository deliberately leaves ordinary engineering judgment to Codex. The Harness focuses on three things Codex should not have to remember manually:

- **Ascend-specific knowledge** — `xingchen-kernel-optimizer`, design experience, bottleneck diagnosis, and optimization patterns.
- **Mechanical evidence state** — task-scoped build/correctness/benchmark/profile records, freshness, and score comparability.
- **High-cost safety checks** — interface/task scope, CANNJudge problem/package identity, secrets, and explicit submission authorization.

`main` is the canonical generic Harness baseline. Task branches carry operator-specific code, tests, scripts, and evidence summaries; generic Harness changes should land on `main` first and then be merged into active task branches.

## Start a task

On an Ascend machine:

```bash
bash scripts/bootstrap_skills.sh
bash scripts/doctor.sh

# First Codex conversation for this task only:
python3 tools/context_state.py bootstrap --task <task> --new-session

# Inspect compact current evidence before trusting an old PASS/score:
python3 tools/task_state.py --task <task>
```

Use a task-scoped config at `config/tasks/<task>.env` when available. A new task can start from `config/agent.env.example`.

The normal evidence loop is:

```bash
python3 tools/agent_loop.py validate  --task <task>
python3 tools/agent_loop.py baseline  --task <task> --name baseline
python3 tools/agent_loop.py diagnose  --task <task>
python3 tools/agent_loop.py candidate --task <task> --name <candidate> --hypothesis '<one mechanism>'
```

`agent_loop.py` writes runtime records under `tasks/<task>/runs/harness/`; those raw records are local/ignored. Keep durable conclusions, retained mechanisms, failed hypotheses worth remembering, and platform facts in the task documentation rather than committing repeated stdout.

For design experience, Codex reads the compact catalog and selects relevant detail itself:

```bash
python3 tools/agent_loop.py design --task <task>
python3 tools/agent_loop.py design --task <task> \
  --select-pattern <pattern-id> \
  --select-pattern <pattern-id>
```

For platform submission:

```bash
python3 tools/agent_loop.py platform --task <task> --submit
```

`--submit` is mandatory. Never store CANNJudge credentials, ciphertext, or private keys in tracked config.

## Evidence semantics

A historical result is not automatically evidence for the current tree.

`tools/evidence_fingerprint.py` binds a run to implementation, relevant gate/case/config context, non-secret Ascend environment identity, and stage commands. `tools/task_state.py` reports old evidence as `fresh`, `stale`, or `unknown`.

Benchmark scores are compared only when their benchmark contexts are compatible. Source changes may make correctness evidence stale without making the previous benchmark baseline incomparable. Platform score evidence is additionally bound to the current official CANNJudge package SHA when platform identity is checked.

## External knowledge

`scripts/bootstrap_skills.sh` installs official Ascend/CANN skills and KDA references into local ignored paths. Generated symlinks are intentionally not tracked, so cloning or using Git linked worktrees does not bake one machine's absolute paths into the repository.

# Xingchenbei Ascend Kernel Agent Harness

A Codex-oriented harness for implementing and optimizing Ascend C competition kernels.

The harness separates five concerns:

1. **Competition contract** — immutable interface, required semantics, hidden-test obligations.
2. **Repository guardrails** — `AGENTS.md` defines what the coding agent may and may not change.
3. **Ascend expertise** — official Ascend Agent Skills are bootstrapped into `.agents/skills/`.
4. **CANNJudge integration** — the official `cannjudge-submit` skill provides problem/package/submission/result/ranking interaction.
5. **Evaluation-driven optimization** — candidates must pass guard → build → correctness before benchmark/profile evidence can promote them.

Current task workspaces:

- `tasks/sparse_flash_attention/`
- `tasks/mhc_expand/`
- `tasks/mhc_sinkhorn/`

## Quick start on an Ascend machine

```bash
cp config/agent.env.example config/agent.env
bash scripts/bootstrap_skills.sh
bash scripts/doctor.sh
python3 tools/agent_loop.py baseline --name baseline
```

Open this repository root as the VS Code workspace when using the Codex extension so repository instructions and local skills are discoverable.

## Core workflow

```text
Codex edits one focused candidate
        ↓
interface guard (optional)
        ↓
build
        ↓
correctness
        ↓
benchmark
        ↓
profile when a concrete question remains
        ↓
promote / reject from measured evidence
```

`tools/agent_loop.py` records every run under `runs/` and tracks the current best score in `runs/best.json`. It never auto-reverts the working tree, so unrelated user changes are not destroyed.

CANNJudge is an optional additional platform-evidence loop. `$cannjudge-submit` may be used to verify current problem/package facts, and—only with explicit user authorization—to submit and query results. Its RSA key/credential material must remain local and untracked.

## External sources

- Official Ascend skills are cloned by `scripts/bootstrap_skills.sh`.
- Official CANN `cann-learning-hub` provides `cannjudge-submit` plus its supporting `ascendc-ops-project` skill.
- MIT Kernel Design Agents is used as optimization methodology/reference, not as NVIDIA expertise for Ascend.
- Huawei-side repository guidance supplied for this project is distilled into the competition-safe rules in root `AGENTS.md`.

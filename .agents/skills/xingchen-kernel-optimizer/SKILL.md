---
name: xingchen-kernel-optimizer
description: Lightweight orchestration for Ascend competition operators: reuse context, protect the contract, use evidence gates, and load design/performance experience only when relevant.
---

# Xingchen Kernel Optimizer

This skill is intentionally small. Let Codex handle ordinary engineering; use the Harness for expensive mistakes and repetitive mechanics.

## Start or resume

At the start of a new Codex conversation for a task:

```bash
python3 tools/context_state.py bootstrap --task <task> --new-session
python3 tools/task_state.py --task <task>
```

Do this once. Stable files already presented in the current session must not be reread unless their hash changed or exact context was lost.

To use another Skill without ritual rereads:

```bash
python3 tools/context_state.py use --task <task> .agents/skills/<skill>/SKILL.md
```

Unchanged -> reuse context. Changed -> consume the diff first.

## Contract and correctness

Authority order is competition/platform evidence -> repository/task contract -> official template -> skills/references -> assumptions.

Keep the public operator interface unchanged unless authoritative evidence permits a change. Keep implementation information internal to tiling/workspace/templates.

For a new or uncertain task, resolve only the facts that can change correctness: ABI, mathematical semantics, supported dtype/shape/modes, target CANN/SOC, and required packaging. Do not force a generic multi-stage design ritual.

Build the simplest contract-complete baseline, then use configured build/validation gates. Never weaken the evaluator to pass.

## Experience lookup

When `tools/ascend_design_analyze.py` exists and design experience is useful:

```bash
python3 tools/ascend_design_analyze.py --task <task>
```

Read the complete summary catalog once. Codex chooses relevant pattern ids; machine suggestions are weak navigation only. Load at most a few detailed patterns for the current question:

```bash
python3 tools/ascend_design_analyze.py \
  --task <task> \
  --select-pattern <id>
```

Do not read long playbooks end-to-end unless a specific unresolved question requires them.

## Performance

Only optimize after correctness evidence exists.

When the branch provides the generic diagnosis mode, use it only when it can answer a concrete question:

```bash
python3 tools/agent_loop.py diagnose --task <task> --name pre-candidate
```

Evidence confidence is `profile-observed > configured hypothesis > static source risk`. Static source risks are not measured bottlenecks.

For each candidate: state one hypothesis, change one major mechanism, build, validate, benchmark on the same cases, and keep/reject from evidence. Profile only when it will choose the next action.

## Platform submission

Use current CANNJudge identity/package evidence before submission when available. Never submit implicitly; explicit user authorization is required.

Local A3/proxy validation and platform 910B evidence must remain clearly distinguished.

## Reporting

Do not spend tokens proving that you followed this skill. Report only new contract facts, conflicts, meaningful design decisions, failed/passed gates, measured performance, and blockers.

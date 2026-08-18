# AGENTS.md

This repository is an AI coding-agent harness for Ascend C competition operators.
The closest nested `AGENTS.md`, when present, overrides this file for that subtree.

## Priority of truth

When instructions conflict, use this order:

1. competition problem statement and platform submission contract;
2. this repository `AGENTS.md` and any closer nested `AGENTS.md`;
3. the official competition template and existing public interface;
4. installed Ascend Agent Skills;
5. KDA-style optimization hypotheses and external references;
6. the agent's own assumptions.

Never change a higher-priority contract to make a lower-priority idea easier to implement.

## Working principles

- Start with `git status --short`; never overwrite unrelated user work.
- Inspect with `rg` / `rg --files` before editing; do not guess paths.
- Keep diffs focused. Avoid unrelated formatting and generated-file noise.
- Build and test claims must be evidence-based. Never report an unexecuted command as passed.
- Correctness has priority over performance.
- Hidden tests exist. Do not optimize only for visible shapes or visible cases.

## External-interface red line

Unless the competition contract explicitly permits it, do not change:

- operator name or registration entrypoint;
- input/output/attribute names, counts, ordering, optionality, defaults, dtype semantics, or shape semantics;
- required submission filenames, directory layout, or externally invoked symbols;
- required functionality, masks, sequence-length semantics, sparse-index semantics, or numerical-output semantics.

Implementation information must stay internal. Shape/SOC/template/tile/core/workspace decisions should be derived by Host Tiling and passed via tiling data, workspace, or compile-time template parameters rather than new public inputs.

## Internal implementation freedom

Within the competition contract, the agent may change:

- Host Tiling strategy and tiling-data fields;
- workspace planning;
- kernel template organization;
- GM/L1/UB/L0A/L0B/L0C planning when supported;
- multicore partitioning;
- data-movement and sparse-gather strategy;
- Matmul/MMAD scheduling;
- Vector/Cube pipeline and buffering;
- softmax / online-softmax implementation;
- precision-safe accumulation and casting strategy.

Do not hardcode one visible shape unless the task contract explicitly states that shape is the entire required domain. Template boundaries must follow real algorithm/resource differences, not evaluator case IDs.

## Ascend Skills

Prefer the installed official Ascend skills for domain-specific decisions:

- `ascendc-operator-design`
- `ascendc-operator-code-gen`
- `ascendc-operator-code-review`
- `ascendc-operator-compile-debug`
- `ascendc-operator-mssanitizer`
- `ascendc-operator-precision-debug`
- `ascendc-operator-performance-eval`
- `ascendc-operator-performance-optim`
- `xingchen-kernel-optimizer` (repository-local orchestration skill)

Do not run generic project-initialization skills inside a competition-provided template unless explicitly requested.

## Required optimization loop

For every meaningful performance candidate:

1. state one focused hypothesis;
2. make one major optimization change at a time;
3. run `scripts/guard.sh` when configured;
4. run `scripts/build.sh`;
5. run `scripts/validate.sh`;
6. only after correctness passes, run `scripts/bench.sh`;
7. run `scripts/profile.sh` when evidence is needed to choose the next hypothesis;
8. record the candidate, commands, result, score, and keep/reject decision;
9. reject or fix any candidate that breaks correctness, even if faster.

Never loosen tolerance, skip required cases, shrink the required range, or alter the reference to manufacture a pass.
Performance claims must come from the configured evaluator/profiler, not Python wall-clock timing unless the competition itself defines wall-clock timing as the metric.

## Task workflow

Before a large implementation:

- read the task contract under `tasks/<task>/TASK.md`;
- inspect the competition statement/template;
- establish a compiling correctness baseline;
- keep `tasks/<task>/design.md` current with dataflow, tiling, memory plan, core split, precision risks, and optimization candidates;
- keep `tasks/<task>/optimization-log.md` evidence-based.

## End-of-task check

Before finishing, verify:

- only intended files changed;
- public interface remains compatible;
- supported dtype/shape/mode/boundary cases remain covered;
- build and correctness commands actually ran, or blockers are stated explicitly;
- performance claims have measured evidence;
- no build outputs, profiler dumps, secrets, machine names, tokens, or environment-specific absolute paths are staged.

# AGENTS.md

This repository is an AI coding-agent harness for Ascend C competition operators.
The closest nested `AGENTS.md`, when present, overrides this file for that subtree.

## Priority of truth

When instructions conflict, use this order:

1. competition problem statement and platform submission contract, including current CANNJudge evidence for the matching problem ID;
2. this repository `AGENTS.md` and any closer nested `AGENTS.md`;
3. the official competition template and existing public interface;
4. installed Ascend Agent Skills;
5. CANNJudge/CANN engineering helper skills and KDA-style optimization references;
6. the agent's own assumptions.

Never change a higher-priority contract to make a lower-priority idea easier to implement.

## Working principles

- Start with `git status --short`; never overwrite unrelated user work.
- Inspect with `rg` / `rg --files` before editing; do not guess paths.
- Keep diffs focused. Avoid unrelated formatting and generated-file noise.
- Build and test claims must be evidence-based. Never report an unexecuted command as passed.
- Correctness has priority over performance.
- Hidden tests exist. Do not optimize only for visible shapes or visible cases.
- Never infer missing operator semantics from identifier names, filenames, or reference-project conventions alone.

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

Prefer the installed official `Ascend/agent-skills` skills for domain-specific decisions:

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

## CANNJudge platform skills

The bootstrap also exposes the official CANN `cann-learning-hub` skills:

- `cannjudge-submit`: platform interaction only — obtain current problem metadata, download/compare the official package, submit code when explicitly authorized, query submission results, and inspect rankings.
- `ascendc-ops-project`: supporting dependency/reference for the CANNJudge skill. For core Ascend C design, implementation, precision, and performance decisions, prefer the dedicated `Ascend/agent-skills` skills above.

Rules for CANNJudge usage:

- Treat a current CANNJudge response/package for the verified matching problem ID as authoritative platform evidence. Do not infer hidden testcase contents.
- Do not overwrite the checked-in competition workspace merely because a freshly downloaded package differs; compare and report the difference first.
- Do not submit to CANNJudge unless the user explicitly requests or authorizes a submission in the current task/session.
- Never ask the user to paste a plaintext CANNJudge password into chat or source files. Follow the skill's RSA-encrypted credential workflow.
- Never print, copy into logs, or commit decrypted passwords, cookies, tokens, RSA private keys, or credential ciphertext.
- Keep generated key material local. `private.pem`, `public.pem`, dependency checkouts, and generated skill symlinks are ignored by Git.
- Do not attempt to discover or access hidden testcases. Use only platform-supported problem, package, submission-result, and ranking interfaces.

## Required development loop

For a new operator, a missing baseline, or a major architecture rewrite, run the generic design pass before substantial kernel edits when task configuration exists:

```bash
python3 tools/agent_loop.py design --task <task> --name initial-design
```

The design pass is advisory. It may combine explicit archetype hints and static source/document signals, but the task contract remains authoritative. Never report a static suggestion as a contract fact.

Before implementing, resolve and record in `tasks/<task>/design.md`:

1. immutable public contract and required domain;
2. mathematical stage/dependency graph;
3. serial dependency axes versus independent axes;
4. logical task/core ownership;
5. physical layout/contiguity and host materialization risks;
6. Host Tiling responsibilities and regime boundaries;
7. aligned/full-tile and tail strategy;
8. register/UB/L1/L0/workspace/GM lifetime plan;
9. storage/compute/accumulator/output precision contract;
10. correctness matrix derived from semantic and hardware boundaries.

The Harness does not choose exact tile sizes, core counts, queue depths, workspace ring depths, TilingKey thresholds, or legal mathematical reformulations. The coding agent should decide them from contract/API/evidence and may reject a design suggestion with a recorded reason.

After the baseline architecture is chosen:

1. implement the simplest contract-complete path;
2. run guard/build/validation;
3. localize failures by contract/tiling/layout/tail/precision/state/synchronization/target API;
4. add minimal regression cases for fixed failures;
5. update `design.md` to match the retained implementation;
6. only then enter performance optimization.

## Required optimization loop

Before the first meaningful performance candidate, run the generic Ascend diagnosis path when available:

```bash
python3 tools/agent_loop.py diagnose --task <task> --name pre-candidate
```

The diagnosis may combine profiler evidence, configured hints and static source risks. Treat them in that order of confidence. Never report a `static_risk_tag` as a measured bottleneck.

For every meaningful performance candidate:

1. state one focused hypothesis tied to the current diagnosis/profile evidence;
2. make one major optimization change at a time;
3. run `scripts/guard.sh` when configured;
4. run `scripts/build.sh`;
5. run `scripts/validate.sh`;
6. only after correctness passes, run `scripts/bench.sh`;
7. run `scripts/profile.sh` or `agent_loop.py profile/diagnose` when evidence is needed to choose the next hypothesis;
8. record the candidate, commands, result, score, diagnosis evidence, and keep/reject decision;
9. reject or fix any candidate that breaks correctness, even if faster.

Never loosen tolerance, skip required cases, shrink the required range, or alter the reference to manufacture a pass.
Performance claims must come from the configured evaluator/profiler, not Python wall-clock timing unless the competition itself defines wall-clock timing as the metric.

## Task workflow

Before a large implementation:

- read the task contract under `tasks/<task>/TASK.md`;
- inspect the competition statement/template;
- use CANNJudge platform evidence to resolve platform-contract ambiguity when available;
- run the generic `design` pass when task configuration exists;
- keep `tasks/<task>/design.md` current with contract, dataflow, task ownership, tiling, memory plan, precision risks and validation matrix;
- establish a compiling correctness baseline;
- keep `tasks/<task>/optimization-log.md` evidence-based once performance work begins.

## End-of-task check

Before finishing, verify:

- only intended files changed;
- public interface remains compatible;
- supported dtype/shape/mode/boundary cases remain covered;
- build and correctness commands actually ran, or blockers are stated explicitly;
- performance claims have measured evidence;
- no build outputs, profiler dumps, secrets, machine names, tokens, or environment-specific absolute paths are staged.

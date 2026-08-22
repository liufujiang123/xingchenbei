---
name: xingchen-kernel-optimizer
description: Contract-first Ascend C competition operator development and evidence-driven optimization with sparse research-derived design/performance guidance.
---

# Xingchen Kernel Optimizer

Coordinate an operator from contract/design through correctness baseline and performance optimization. Correctness and the platform-visible interface are always higher priority than performance.

## Sources of truth

Use this order:

1. nearest `AGENTS.md`;
2. task statement/template and current verified platform evidence;
3. target CANN/SOC build and correctness results;
4. measured benchmark/profile evidence;
5. official Ascend skills and this repository's design/performance pattern libraries.

Use `cannjudge-submit` only for platform facts/actions. Never submit without explicit user authorization and never expose credentials.

Use the official Ascend skills for architecture/tiling, code generation, compile/debug, precision, performance evaluation and optimization. This skill coordinates them; it does not replace them.

# Part I — Operator development

Use for a new operator, missing baseline, or major architecture rewrite. Do not force a full redesign for a small bug fix.

Read:

- `docs/ascend-operator-development.md`
- `tasks/<task>/TASK.md`
- official statement/template/package
- `tasks/<task>/design.md` when present

Start with:

```bash
python3 tools/agent_loop.py design \
  --task <task> \
  --name initial-design
```

After reading the contract, Codex may add broad archetype hints:

```bash
python3 tools/agent_loop.py design \
  --task <task> \
  --archetype reduction \
  --name initial-design
```

Allowed archetypes: `elementwise`, `broadcast`, `reduction`, `scan`, `recurrent`, `sparse`, `gather`, `matmul`, `normalization`, `attention`, `composite`.

Declared archetypes are hints, not contract facts. Static suggestions are weaker still. Never infer semantics from filenames, identifiers, or another project's conventions.

## Sparse knowledge injection

The design library is intentionally broader than the context shown to Codex.

Normal `design` output must obey this attention budget:

- inject at most **3 active design patterns**;
- prefer current archetype/source/document signals over generic checklist items;
- keep the active set phase-diverse where practical;
- allow at most **1 risk-triggered deep dive** with validation detail;
- keep other relevant patterns as **ids only**;
- do not treat deferred ids as obligations.

The purpose is to ask the highest-value questions at the right time, not to make Codex execute every rule.

If a concrete risk or unknown needs more detail, expand one pattern explicitly:

```bash
python3 tools/ascend_design_analyze.py \
  --task <task> \
  --expand-pattern <pattern-id>
```

Do not expand the whole catalog. A pattern is advisory and Codex may reject it with a reason tied to contract, target API, resource budget, build/correctness evidence, or measured behavior.

## Design decisions Codex must eventually resolve

The complete design should cover these topics in `tasks/<task>/design.md`, but they do **not** all need to be injected in one prompt:

- immutable contract and legal dtype/shape/mode/optional domain;
- mathematical stage/dependency graph;
- dependency axes versus independent axes;
- logical task/core ownership before micro-tiling;
- physical layout, strides and host-materialization risks;
- Host Tiling responsibilities and genuine regime boundaries;
- aligned full-tile and partial-tail handling;
- register/UB/L1/L0/workspace/GM lifetime plan;
- storage/compute/accumulator/output precision semantics;
- correctness matrix from semantic and hardware boundaries.

## High-value architecture prompts

Apply only when the contract/source supports them:

- **elementwise/broadcast** — choose independent tile axis and broadcast staging; establish a simple GM→UB→V→GM correctness path before deeper pipelining.
- **reduction/normalization** — define reduction ownership, merge cost, accumulator semantics and baseline full-data pass count.
- **scan/recurrent** — keep carried state with its dependency chain when practical; parallelize orthogonal axes.
- **sparse/gather/paged** — define index/page/chunk semantics and output order before locality/coalescing decisions.
- **Cube/matmul** — choose M/N/K ownership and resident-versus-streaming operand roles under L1/L0 constraints.
- **mixed Cube+Vector** — define producer, consumer, intermediate location, true-ready edge, reuse edge and safe in-flight distance before flags/rings/stage counts.

Do not copy fixed tiles, stage counts, ring depths, synchronization intervals, transfer thresholds or AIC:AIV ratios from reference projects.

## Codex autonomy

The Harness must not choose exact mathematical reformulations, tile sizes, core counts, queue depths, ring depths, UB/L1/L0 layouts, TilingKey counts/thresholds, or specializations. Codex owns those choices and must justify them from the current operator and target.

The design layer succeeds when it exposes a high-value missing decision early; it is not intended to make every operator structurally identical.

## Baseline completion gate

Before performance work:

- public interface unchanged;
- target build passes;
- required correctness/precision matrix passes;
- any local target adaptation is scoped and auditable;
- `tasks/<task>/design.md` matches the retained architecture.

# Part II — Performance optimization

Read:

- `docs/ascend-optimization-playbook.md`
- `docs/ascend-kernel-research.md`
- `tasks/<task>/optimization-log.md` when present

Before choosing a candidate, prefer:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --name pre-candidate
```

With fresh build/correctness evidence, a cheap source-only pass may use `--skip-build --skip-validate --skip-profile`.

Evidence priority is:

`profile-observed > configured hypothesis > static source risk`.

Never report a static risk as a measured bottleneck.

## Performance model

Classify the hot path as one primary family:

- `vector`: `GM -> UB -> V -> UB -> GM`
- `cube`: `GM -> L1 -> L0 -> Cube -> L0C/FIX -> GM`
- `mixed_cv`: substantial Cube and Vector stages exchange tiles/workspace

For the relevant resources, model Scalar, MTE1/MTE2/MTE3, Vector, Cube, UB/L1/L0, workspace and synchronization edges.

For each wait/barrier ask:

- true data dependency or buffer/workspace reuse?
- how much work may safely be in flight?
- which values remain live across that lead distance?

## Research-derived performance questions

Use only when supported by evidence:

- keep recurrence chains local and parallelize orthogonal axes;
- size rings/stages from live ranges, async hazards and cache/on-chip capacity;
- improve cache reuse or phase-shift independent traversal when cores contend for the same GM phase;
- assemble sparse/paged fragments directly in UB/L1 when legal instead of materializing GM intermediates;
- reduce full GM passes when an online/pass-fused formulation preserves numerical semantics;
- verify anomalously slow dtype/shape/API combinations use the intended hardware path;
- use a small hardware-pruned regime search rather than blind autotuning.

High-value lessons from production kernels:

- more buffers/stages are not monotonically better;
- only asynchronously shared buffers need multi-buffering;
- ping-pong can still leave output-block boundary bubbles;
- resident and streaming operands are asymmetric;
- per-task C/V ready/wait can create lockstep;
- synchronization may sometimes be batched, but true dependencies and tail flush remain mandatory;
- task order affects both cache reuse and synchronized GM contention;
- reducing GM traffic can justify extra Vector arithmetic;
- MicroAPI/register kernels are advanced tools for a proven hotspot, not a default rewrite.

## One-candidate rule

Each candidate states:

- hypothesis;
- evidence level and bottleneck;
- expected Ascend resource/pipeline effect;
- one major mechanism changed;
- added UB/L1/workspace/code-size cost;
- correctness/precision risk;
- exact same-case evaluation plan.

Then run target build → correctness → benchmark. Profile only to answer a concrete question. Do not stack an unproven mechanism into the retained implementation.

When mechanisms interact, prefer this order:

1. dependency-safe task decomposition;
2. layout/data movement;
3. buffering/residency/pipeline;
4. synchronization/window tuning;
5. tiling/regime autotune;
6. advanced hardware-path/register microkernel.

## Promotion

Promote only when:

- public interface unchanged;
- target build passes;
- required correctness/precision passes;
- same-case performance improves beyond noise;
- no required shape/dtype/mode domain is narrowed;
- target/proxy evidence is labeled correctly.

Record `PROMOTE`, `REJECT` or `INCONCLUSIVE`, including failed experiments.

CANNJudge score is authoritative platform evidence only when actually returned by CANNJudge. Local proxy measurements are not target-platform proof. Never optimize by guessing hidden testcases.

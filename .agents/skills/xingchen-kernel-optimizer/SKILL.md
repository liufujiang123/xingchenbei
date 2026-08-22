---
name: xingchen-kernel-optimizer
description: Evidence-driven Ascend C competition operator development and optimization with contract-first design, source/profile diagnosis and research-derived architecture, memory, scheduling, synchronization and hardware-path reasoning.
---

# Xingchen Kernel Optimizer

Use this skill to coordinate a competition operator from contract/design through correctness baseline and performance optimization. Correctness and the public interface remain higher priority than performance.

## Sources of truth

1. nearest `AGENTS.md`;
2. task statement/template and current platform evidence;
3. target CANN/SOC build and correctness results;
4. measured benchmark/profile evidence;
5. official Ascend skills and this repository's compact design/performance pattern libraries.

Use `cannjudge-submit` only for CANNJudge facts/actions and never submit without explicit user authorization. Never expose credentials.

Use the official Ascend skills for architecture/tiling, code generation, compile/debug, precision, performance evaluation and optimization. This skill coordinates them; it does not replace them.

# Part I — Operator development

Use this section for a new operator, a missing baseline, or a major architecture rewrite. For a small bug fix, do not force a full redesign.

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

If the contract has already been read and the mathematical archetype is clear, Codex may provide explicit hints:

```bash
python3 tools/agent_loop.py design \
  --task <task> \
  --archetype reduction \
  --archetype broadcast \
  --name initial-design
```

Allowed archetypes are intentionally broad: `elementwise`, `broadcast`, `reduction`, `scan`, `recurrent`, `sparse`, `gather`, `matmul`, `normalization`, `attention`, `composite`.

`declared_archetypes` are Codex/user hints, not a replacement for the task contract. `suggested_archetypes` come from conservative source signals and are even weaker. Never infer semantics from a filename or identifier alone.

## Mandatory baseline design decisions

Before substantial kernel code, resolve and record these decisions in `tasks/<task>/design.md`:

1. **Contract** — immutable inputs/outputs/attrs/defaults/modes, shape/dtype domain, target CANN/SOC and submission files.
2. **Semantic graph** — mathematical stages, intermediates, reductions, masks and persistent state.
3. **Dependency axes** — independent vs reduction-coupled vs recurrence-coupled vs producer-consumer work.
4. **Task ownership** — what one logical core task owns before micro-tiling.
5. **Physical layout** — contiguous axes/strides and whether host transforms are views or materializations.
6. **Host Tiling** — block count, loops, tile regimes, internal mode keys and workspace passed through TilingData.
7. **Tail/alignment** — full-tile path and partial-tile path.
8. **Memory lifetime** — register/UB/L1/L0/workspace/GM placement, state vs cross-stage exchange vs scratch.
9. **Precision** — storage/compute/accumulator/output dtype, reduction order and padding identities.
10. **Validation matrix** — modes, dtypes, optionals, smallest shapes, alignment boundaries, tails and stress/generalization cases.

These are questions to answer, not fixed implementation templates.

## Architecture prompts by operator family

Apply only when the contract/source supports the family:

- **elementwise/broadcast**: choose independent tile axis and broadcast staging; start with a simple GM->UB->V->GM correctness skeleton.
- **reduction/normalization**: define reduction ownership and accumulator semantics; count full-data passes and merge cost.
- **scan/recurrent**: keep carried state with its dependency chain when practical; parallelize orthogonal axes.
- **sparse/gather/paged**: define index/page/chunk semantics and output order before optimizing staging/locality.
- **Cube/matmul**: choose M/N/K ownership and resident-vs-streaming operand roles under L1/L0 constraints.
- **mixed Cube+Vector**: define producer, consumer, intermediate location, true-ready edge, reuse edge and safe in-flight distance before selecting flags/ring depth.

Do not copy fixed tiles, stage counts, ring depths, synchronization intervals or AIC:AIV ratios from reference projects.

## Codex autonomy

The design library deliberately does not choose exact tile sizes, core counts, queue depths, TilingKey thresholds, buffer layouts or legal mathematical reformulations. Codex owns those choices and may reject a Harness suggestion with a reason tied to contract/API/evidence.

The design mode is successful when it exposes missing decisions early; it is not intended to make every operator structurally identical.

## Baseline completion gate

Before entering performance work:

- public interface unchanged;
- target build passes;
- required correctness/precision matrix passes;
- local target adaptation, if any, is scoped and auditable;
- `tasks/<task>/design.md` reflects the retained architecture.

Only then continue to Part II.

# Part II — Performance optimization

Read:

- `docs/ascend-optimization-playbook.md`
- `docs/ascend-kernel-research.md`
- `tasks/<task>/optimization-log.md` when present

Record `git status --short` and identify the immutable platform-visible interface.

## Diagnosis is the default entry point

Before choosing a performance candidate, prefer:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --name pre-candidate
```

This runs configured gates, runs `PROFILE_CMD` when available, performs conservative source analysis, and attaches a ranked research-derived shortlist to the task-local harness record.

When recent build/correctness evidence already exists and a cheap source-only pass is intended:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --skip-build \
  --skip-validate \
  --skip-profile \
  --name source-scan
```

`profile` mode also attaches diagnosis automatically.

Do not treat `static_risk_tags` as measured bottlenecks. Evidence priority is:

`profile-observed > configured hypothesis > static source risk`.

If profile and static class inference disagree, inspect the conflict before editing.

## Stable profiler evidence contract

A task profiler wrapper may optionally emit:

```text
HARNESS_OPERATOR_CLASS=<vector|cube|mixed_cv>
HARNESS_BOTTLENECKS=<comma-separated tags>
HARNESS_PROFILE_NOTE=<short measured observation>
```

Allowed tags include:

`pipeline`, `memory`, `bandwidth`, `cache`, `compute`, `latency`, `underutilization`, `scalar`, `synchronization`, `tiling`, `sparse`.

Only emit claims supported by actual profiler evidence.

Generic task configuration knobs:

```bash
PERF_OPERATOR_CLASS=auto
PERF_SOURCE_DIRS=''
PERF_BOTTLENECK_HINTS=''
PERF_PLAN_LIMIT=5
PERF_ADVANCED=0
```

Do not encode testcase IDs or reference-project magic numbers in these settings.

## Mandatory Ascend performance model

### A. Classify the hot path

Resolve one primary class:

- `vector`: `GM -> UB -> V -> UB -> GM`
- `cube`: `GM -> L1 -> L0 -> Cube -> L0C/FIX -> GM`
- `mixed_cv`: substantial Cube and Vector stages exchange tiles/workspace

Do not force C/V techniques onto pure Vector work.

### B. Draw resources and true dependencies

List relevant Scalar, MTE1/MTE2/MTE3, Vector, Cube, UB/L1/L0, workspace and flag edges.

For every wait/barrier ask:

- true data dependency or only buffer/workspace reuse?
- how many tasks may safely be in flight?
- which values remain live across that lead distance?

### C. Perform the research-derived scan

Check these questions before selecting a candidate:

1. **Dependency axes** — keep true recurrence chains local; parallelize orthogonal axes.
2. **Working-set liveness** — derive buffer/ring depth from live ranges, async hazards and cache/on-chip capacity.
3. **Locality/conflict** — consider grouped/swizzled reuse and phase-shifted traversal when cores contend for the same GM phase.
4. **Movement** — assemble sparse/paged/reformatted fragments directly in UB/L1 when legal instead of materializing GM intermediates.
5. **Algorithmic passes** — fuse related full reduction scans when an online correction preserves the numerical contract.
6. **Hardware path** — verify anomalously slow dtype/shape/API combinations lower to the intended hardware path.
7. **Autotune regime** — use a small hardware-pruned search keyed by performance-relevant shape/dtype/layout regimes.

## High-value rules learned from real Ascend kernels

- More buffers/stages are not monotonically better.
- Only asynchronously shared buffers need multi-buffering.
- Ping-pong can still leave output-block boundary bubbles; model prologue/main/drain.
- Resident and streaming operands are asymmetric.
- Per-task C/V ready/wait can create lockstep; a bounded credit window may be better when dependency distance permits.
- Cross-core synchronization can sometimes be batched, but tail flush and true dependencies are mandatory.
- Ring depth should follow temporary live windows and cache fit, not a fixed maximum.
- Task order can affect both cache reuse and synchronized GM contention.
- Recurrent state should stay on one task/core when possible.
- Reducing GM passes can be worth extra Vector arithmetic.
- Supported dtype does not imply a fast implementation path.
- MicroAPI/register kernels are advanced tools for a proven inner hotspot, not a default rewrite.

Never copy fixed stage counts, ring depths, synchronization intervals, AIC:AIV ratios, tile shapes or transfer thresholds from another repository.

## Candidate planner

`agent_loop.py diagnose/profile` already embeds a shortlist. Standalone lookup remains available:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class <vector|cube|mixed_cv> \
  --bottleneck <tag>
```

Advanced patterns stay hidden unless profile and target API evidence justify them:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --advanced-diagnosis
```

The registry is a hypothesis library, not an automatic rewrite engine.

## One-candidate rule

Each candidate must state:

- hypothesis;
- evidence level and bottleneck;
- expected Ascend resource/pipeline effect;
- one major mechanism changed;
- added UB/L1/workspace/code-size cost;
- correctness/precision risk;
- exact same-case evaluation plan.

Then run target build -> correctness -> benchmark. Profile only to answer a concrete question. Do not stack an unproven mechanism into the retained implementation.

When candidates interact, use this order:

1. dependency-safe task decomposition;
2. layout/data movement;
3. buffering/residency/pipeline;
4. synchronization/window tuning;
5. tiling/regime autotune;
6. advanced hardware-path/register microkernel.

Recompute memory budgets after any change to live buffers.

## Promotion

Promote only when:

- public interface unchanged;
- target build passes;
- required correctness/precision passes;
- same-case performance improvement exceeds noise;
- no required shape/dtype/mode is narrowed;
- target/proxy evidence is labeled correctly.

Record `PROMOTE`, `REJECT` or `INCONCLUSIVE` in `tasks/<task>/optimization-log.md`, including failed experiments.

CANNJudge score is authoritative platform evidence only when actually returned by CANNJudge. Local proxy measurements are not target-platform proof.

Never optimize by guessing hidden testcases.

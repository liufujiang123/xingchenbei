---
name: xingchen-kernel-optimizer
description: Evidence-driven Ascend C competition optimization with research-derived pipeline, memory, scheduling, synchronization and hardware-path reasoning.
---

# Xingchen Kernel Optimizer

Use this skill after the competition contract is understood. Correctness and the public interface remain higher priority than performance.

## Sources of truth

1. nearest `AGENTS.md`;
2. task statement/template and current platform evidence;
3. target CANN/SOC build and correctness results;
4. measured benchmark/profile evidence;
5. official Ascend skills and this repository's research-derived pattern library.

Use `cannjudge-submit` only for CANNJudge facts/actions and never submit without explicit user authorization. Never expose credentials.

Use the official Ascend skills for architecture/tiling, code generation, compile/debug, precision, performance evaluation and performance optimization. This skill coordinates them; it does not replace them.

## Before performance work

Confirm a correctness baseline exists. Read:

- `docs/ascend-optimization-playbook.md`
- `docs/ascend-kernel-research.md`
- `tasks/<task>/optimization-log.md` when present

Record `git status --short` and identify the immutable platform-visible interface.

## Mandatory Ascend performance model

Before editing performance code, write a short model in the task log.

### A. Classify the hot path

Choose one primary class for the candidate:

- `vector`: `GM -> UB -> V -> UB -> GM`
- `cube`: `GM -> L1 -> L0 -> Cube -> L0C/FIX -> GM`
- `mixed_cv`: substantial Cube and Vector stages exchange tiles/workspace

Do not force C/V techniques onto pure Vector work.

### B. Draw resources and true dependencies

List relevant Scalar, MTE1/MTE2/MTE3, Vector, Cube, UB/L1/L0, workspace and flag edges.

For every wait/barrier ask:

- is this a true data dependency?
- or only buffer/workspace reuse?
- how many tasks may safely be in flight before overwrite or semantic dependency?

### C. Perform the research-derived scan

Check these six questions before choosing a candidate:

1. **Dependency axes** — which axis must remain serial, and which orthogonal axes can be parallelized?
2. **Working-set liveness** — which buffers are simultaneously live? Which buffers are actually touched asynchronously? Does a ring/stage count fit UB/L1/L2 rather than merely fit correctness?
3. **Locality/conflict** — can task order improve cache reuse, or are many cores reading the same GM region at the same phase?
4. **Movement** — can paged/sparse/reformatted data be assembled directly in UB/L1 instead of materialized in GM? Are repeated small transfers worth batching?
5. **Hardware path** — does the target dtype/shape/API lower to the intended vector/Cube/DMA path, or silently use a scalar/slow conversion path?
6. **Algorithmic passes** — can repeated full scans/reductions be fused while preserving the numerical contract?

### D. Diagnose the bottleneck

Use source evidence plus `ascendc-operator-performance-eval`/msprof or equivalent profiling where available. Tag with one or more of:

`pipeline`, `memory`, `bandwidth`, `cache`, `compute`, `latency`, `underutilization`, `scalar`, `synchronization`, `tiling`, `sparse`.

Prefer a measured pipeline gap or resource symptom over intuition.

## Candidate planner

Generate a small shortlist:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class <vector|cube|mixed_cv> \
  --bottleneck <tag>
```

The default output contains only `core` patterns. Keep this default for ordinary optimization.

Use:

```bash
python3 tools/ascend_perf_plan.py ... --advanced
```

only when profiling and target CANN/SOC API evidence justify lower-level or more fragile mechanisms such as MicroAPI register kernels or direct C/V handoff.

The registry is a hypothesis library, not an automatic rewrite engine.

## High-value rules learned from real Ascend kernels

- **More buffers/stages are not monotonically better.** Derive stage/ring depth from live ranges, dependency distance, on-chip capacity and cache working set.
- **Only buffers participating in asynchronous producer/consumer access need multi-buffering.** Do not double-buffer Vector-only temporaries without a hazard.
- **Ping-pong may still leave block-boundary bubbles.** Model prologue/main/epilogue and consider cross-block preload when MTE2 gaps remain.
- **Resident and streaming operands are asymmetric.** A reused resident operand usually should not consume the same multi-buffer budget as the streamed side.
- **Cross-core ready/wait per tile can create lockstep.** For mixed C/V, consider legal synchronization batching or a bounded credit window; derive the window from true dependency distance and storage capacity.
- **Task order affects memory behavior.** Group/swizzle for reuse and phase-shift independent traversal when simultaneous same-address traffic causes bandwidth conflict.
- **A recurrence should stay local when possible.** Parallelize independent sequence/head/row axes rather than synchronizing recurrent state across cores.
- **Avoid GM intermediates.** Fuse fragmented gather/reformat into on-chip staging when transfer granularity remains efficient.
- **Supported dtype does not guarantee a fast hardware path.** Inspect generated target code or API branch when a dtype/shape is anomalously slow.
- **Reduce bytes before adding arithmetic.** Online/pass-fused formulations can win when they remove full GM scans and remain numerically valid.

Do not copy fixed values such as stage count, ring depth, synchronization interval, AIC:AIV ratio, tile shape or transfer threshold from another repository.

## One-candidate rule

Each candidate must state:

- hypothesis;
- observed bottleneck;
- expected resource/pipeline effect;
- one major mechanism changed;
- additional UB/L1/workspace or code-size cost;
- correctness/precision risk;
- exact same-case evaluation plan.

Then execute target build -> correctness -> benchmark; profile only to answer a concrete question.

Do not stack an unproven candidate into the retained implementation.

## Ordering

When candidates interact, order them explicitly:

1. semantic/dependency-safe task decomposition;
2. layout/data-movement changes;
3. buffering/residency/pipeline changes;
4. synchronization/window tuning;
5. tile/stage/autotune parameters;
6. advanced hardware-path or register microkernels.

Recompute memory budgets after a change that alters live buffers.

## Promotion

Promote only when:

- public interface unchanged;
- target build passes;
- required correctness/precision passes;
- same-case performance improvement exceeds noise;
- no required shape/dtype/mode is narrowed;
- target/proxy evidence is labeled correctly.

Record `PROMOTE`, `REJECT` or `INCONCLUSIVE` in `tasks/<task>/optimization-log.md`, including failed experiments.

CANNJudge score is authoritative platform evidence only when actually returned by CANNJudge. Local A3 measurements must not be presented as 910B proof.

## Attention/sparse-specific extension

For attention-like or sparse kernels also inspect:

- sparse/paged gather coalescing and direct on-chip staging;
- online softmax / pass fusion;
- state/recurrent dependency placement;
- Q/K/V residency and reuse asymmetry;
- Cube/Vector producer-consumer overlap;
- workspace live windows and cross-core synchronization;
- Matmul/MMAD utilization;
- FP32-sensitive accumulation;
- avoidance of large score/state GM intermediates.

Never optimize by guessing hidden testcases.

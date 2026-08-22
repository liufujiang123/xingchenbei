---
name: xingchen-kernel-optimizer
description: Run an evidence-driven Ascend C competition-kernel workflow with Ascend-first architecture, pipeline, memory and multicore optimization reasoning.
---

# Xingchen Kernel Optimizer

Use this skill for competition-kernel implementation and optimization inside this repository.

## Before changing code

1. Read the nearest `AGENTS.md`.
2. Read `tasks/<task>/TASK.md` and the authoritative competition statement/template.
3. Run `git status --short`.
4. Inspect the existing operator with `rg` / `rg --files`.
5. Identify the immutable platform-visible interface.
6. If the platform contract is ambiguous and CANNJudge access is available, use `cannjudge-submit` to obtain current problem/package evidence before guessing.
7. If no correctness baseline exists, prioritize that before performance work.

## Platform routing

Use `cannjudge-submit` for CANNJudge-specific work. Do not use platform tooling to infer or access hidden testcases. Never ask for a plaintext CANNJudge password or echo credential material.

## Ascend domain routing

Use official Ascend skills as appropriate:

- architecture/dataflow/tiling design -> `ascendc-operator-design`
- Host/Kernel code generation and implementation -> `ascendc-operator-code-gen`
- compiler errors -> `ascendc-operator-compile-debug`
- numerical failures -> `ascendc-operator-precision-debug`
- measured performance analysis -> `ascendc-operator-performance-eval`
- optimization hypotheses/implementation -> `ascendc-operator-performance-optim`

Official skills are advisors; repository and competition contracts remain higher priority.

## Baseline phase

The baseline goal is the simplest implementation that preserves the external interface, covers the required dtype/shape/mode domain, builds on the target CANN/SOC environment, and passes the configured correctness evaluator.

Do not start speculative performance work until build and correctness pass.

## Mandatory Ascend-first performance model

Before proposing a performance code change, read `docs/ascend-optimization-playbook.md` and write down the hot-path model in the task notes/log.

### 1. Classify the hot path

Choose exactly one primary class for the candidate:

- `vector`: SIMD/Vector dominated; typical flow `GM -> UB -> V -> UB -> GM`.
- `cube`: Cube/matrix dominated; typical flow includes `L1/L0A/L0B -> M -> L0C/FixPipe`.
- `mixed_cv`: substantial Cube and Vector stages communicate through on-chip buffers/workspace.

Do not suggest C/V parallelism for a pure Vector operator. For Vector work, inspect MTE2/MTE3 versus V overlap instead.

### 2. Build a resource/dependency graph

Identify, as applicable:

- Scalar/control work;
- MTE1/MTE2/MTE3 movement;
- Vector queue work;
- Cube queue work;
- UB/L1/L0 buffers;
- workspace producer/consumer slots;
- true SetFlag/WaitFlag/PipeBarrier dependencies;
- false serialization caused only by buffer reuse.

For each stage, ask whether adjacent tiles can execute on independent queues at the same time.

### 3. Diagnose the bottleneck

Use source evidence plus `ascendc-operator-performance-eval`/profiling where available. Classify the current issue with tags such as `pipeline`, `memory`, `bandwidth`, `compute`, `latency`, `underutilization`, or `scalar`.

Prefer measured pipeline gaps/instruction utilization over intuition.

### 4. Generate Ascend-specific candidates

Query the repository registry after classification, for example:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class vector \
  --bottleneck pipeline \
  --bottleneck memory
```

or:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class mixed_cv \
  --bottleneck pipeline
```

The registry is a hypothesis library, not an automatic rewrite engine. Confirm target CANN/SOC API support before coding.

## Ascend optimization ladder

Consider these families only when the performance model says they apply.

### Vector operators

- MTE2/MTE3 <-> V overlap;
- TPipe/TQue producer-consumer pipeline;
- double buffer / ping-pong buffers when UB permits;
- aligned full-tile DataCopy fast paths;
- UB operand/accumulator reuse;
- multicore/task-granularity balance;
- remove scalar hot-loop div/mod and invariant address work;
- precision-safe reduction specialization.

### Cube operators

- Cube-friendly M/N/K tiles and utilization;
- L1/L0 reuse;
- MTE1/MTE2 <-> Cube overlap;
- ping-pong matrix staging;
- FixPipe/output overlap where legal;
- multicore tiling and tail balance.

### Mixed Cube + Vector operators

Treat Cube and Vector as a producer-consumer pipeline, not automatically sequential stages. Investigate:

- `C(tile n+1) || V(tile n)` overlap;
- removing unnecessary cross-pipeline barriers;
- workspace ping-pong/ring buffering so Cube does not wait for Vector to release the same slot;
- AIC/AIV work-ratio tuning when profile evidence shows an asymmetric bottleneck;
- Vector-side double buffering;
- internal fusion/avoiding GM materialization of Cube intermediates when the public contract permits.

Do not copy fixed ratios or buffer counts from another operator. The GroupedMatmul best-practice pattern is evidence that these mechanisms can matter, not a universal parameter choice.

## Candidate phase

For each candidate:

1. State one concrete hypothesis tied to a measured/resource-model bottleneck.
2. State the expected pipeline/resource effect, e.g. “hide MTE2 of tile n+1 behind V of tile n” or “remove false C/V dependency by using independent workspace slots”.
3. Change one major optimization dimension at a time.
4. Keep the public interface unchanged.
5. Run target build and configured correctness before performance measurement.
6. Benchmark the same case matrix with warmup and repeated batches.
7. Keep/reject/inconclusive based on evidence; do not stack an unproven candidate into the baseline.

A performance candidate is incomplete if it reports only latency without explaining which Ascend resource/pipeline behavior changed.

## Profile phase

Use profiling only to answer a concrete question, such as:

- Is Vector waiting on MTE2 or MTE3?
- Is Cube waiting on MTE1/L1 staging?
- Are C and V serialized by a true dependency or merely workspace reuse?
- Is usedCoreNum below useful hardware parallelism for this shape class?
- Did double buffering reduce pipeline gaps enough to justify its UB cost?

Interpret profiler results with the official Ascend performance skills. Do not translate NVIDIA-specific metrics mechanically to Ascend.

## Promotion rule

A candidate may be promoted only when build and correctness pass, the same-case benchmark improves beyond measurement noise, the required functional domain is not narrowed, precision remains within contract, and SOC-specific proxy evidence is labeled correctly.

Record the measured result, pipeline hypothesis and decision in `tasks/<task>/optimization-log.md`. Record rejected/inconclusive candidates too.

If CANNJudge is used as an additional evaluator, record submission ID and returned status/score evidence without recording credentials.

## Attention-like sparse kernels

For attention-like sparse kernels, additionally consider sparse gather coalescing, Q sequence x head x sparse-index multicore partition, sparse tile size/tails, GM/L1/UB residency, Cube/Vector overlap, Matmul/MMAD utilization, stable/online softmax, FP32-sensitive accumulation and avoiding large score intermediates.

Do not apply a technique merely because it helps CUDA/Triton kernels; confirm it maps to the target Ascend architecture/API.

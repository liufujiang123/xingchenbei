---
name: xingchen-kernel-optimizer
description: Run an evidence-driven Ascend C competition-kernel workflow. Use for establishing a correctness baseline, planning one focused optimization candidate, running build/validate/bench/profile gates, and deciding keep/reject without changing the competition interface.
---

# Xingchen Kernel Optimizer

Use this skill for competition-kernel implementation and optimization inside this repository.

## Before changing code

1. Read the nearest `AGENTS.md`.
2. Read `tasks/<task>/TASK.md` and the authoritative competition statement/template.
3. Run `git status --short`.
4. Inspect the existing operator with `rg` / `rg --files`.
5. Identify the immutable platform-visible interface.
6. If no correctness baseline exists, prioritize that before performance work.

## Ascend domain routing

Use official Ascend skills as appropriate:

- architecture/dataflow/tiling design → `ascendc-operator-design`
- Host/Kernel code generation and implementation → `ascendc-operator-code-gen`
- compiler errors → `ascendc-operator-compile-debug`
- numerical failures → `ascendc-operator-precision-debug`
- measured performance analysis → `ascendc-operator-performance-eval`
- optimization hypotheses/implementation → `ascendc-operator-performance-optim`

Official skills are advisors; repository and competition contracts remain higher priority.

## Baseline phase

The baseline goal is the simplest implementation that:

- preserves the external interface;
- covers the required dtype/shape/mode domain;
- builds on the target CANN/SOC environment;
- passes the configured correctness evaluator.

Run:

```bash
python3 tools/agent_loop.py baseline --name baseline
```

Do not start speculative performance work until build and correctness pass.

## Candidate phase

For each optimization candidate:

1. Write one concrete hypothesis, e.g. “aggregate adjacent sparse indices to reduce GM transfer setup overhead”.
2. Change one major optimization dimension at a time.
3. Keep the public interface unchanged.
4. Execute:

```bash
python3 tools/agent_loop.py candidate \
  --name <candidate-name> \
  --hypothesis "<single focused hypothesis>"
```

The loop enforces guard → build → validate → benchmark. A failed guard/build/validation candidate is rejected before benchmark.

## Profile phase

Use profiling only to answer a concrete question:

```bash
python3 tools/agent_loop.py profile \
  --name <candidate-name>-profile \
  --hypothesis "<question the profile should answer>"
```

Interpret profiler results with the official Ascend performance skills. Do not translate NVIDIA-specific metrics mechanically to Ascend.

## Promotion rule

A candidate may be promoted only when:

- build passed;
- required correctness passed;
- benchmark score was parsed from the official/configured evaluator;
- it beats the current retained implementation under `BENCH_DIRECTION`;
- it does not narrow the required functional domain.

Record the measured result and decision in `tasks/<task>/optimization-log.md`.

## Search dimensions for attention-like sparse kernels

Consider, based on evidence:

- sparse gather coalescing / contiguous-run aggregation;
- Q sequence × head × sparse-index multicore partition;
- sparse tile size and tail handling;
- GM/L1/UB residency and buffering;
- Cube/Vector overlap and synchronization;
- Matmul/MMAD shape utilization;
- stable softmax / online softmax;
- FP16/BF16 input with FP32-sensitive accumulation where appropriate;
- avoiding materialization of large score intermediates.

Do not apply a technique merely because it helps CUDA/Triton kernels; confirm it maps to the target Ascend architecture/API.

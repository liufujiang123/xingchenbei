# Ascend operator development playbook

This document covers the **pre-performance** half of the Harness: turning a competition contract into a correct, auditable Ascend implementation. It is deliberately a decision framework rather than a cookbook. The concrete algorithm, tile sizes, core count, buffer depths, TilingKey split and kernel structure remain Codex decisions unless the contract or target API forces them.

The intended flow is:

`contract -> semantic graph -> dependency/task ownership -> layout -> tiling -> memory/precision -> operator-family architecture -> correctness baseline -> performance diagnosis`

Use the official Ascend design/code-generation/debug skills for API details. This repository layer keeps the reasoning consistent across tasks.

## 1. Start from evidence, not names

Before implementation, read the task statement, official template/package and nearest `AGENTS.md`. Treat identifier names, existing stubs and reference-project conventions as hints only.

Do not infer missing semantics from names such as `backward`, `mask`, `state`, `score`, `index`, or a filename. If two sources appear to disagree, keep the ambiguity visible until the higher-priority source resolves it.

For a new or substantially redesigned operator, run:

```bash
python3 tools/agent_loop.py design \
  --task <task> \
  --name initial-design
```

If Codex has already classified the mathematical archetype from the contract, it may add one or more explicit hints:

```bash
python3 tools/agent_loop.py design \
  --task <task> \
  --archetype reduction \
  --archetype broadcast \
  --name initial-design
```

The output is advisory. `declared_archetypes` are explicit hints; `suggested_archetypes` come from conservative source signals and are never contract facts.

## 2. Freeze the public contract

Write down only facts supported by the authoritative sources:

- operator/registration entrypoint;
- input/output order, optionality and shape/dtype semantics;
- attrs, defaults and mode semantics;
- required filenames and symbols;
- target CANN/SOC facts;
- legal shape/rank/mode domain.

Implementation choices such as tile size, core count, workspace layout, queue depth and specialization stay internal. Pass them through Host Tiling/TilingData, workspace or compile-time implementation mechanisms rather than changing the public interface.

When multiple signals describe the same mode (for example shape rank plus an optional attr), decide which is the semantic source of truth and what agreement is required. Do not let Host and Kernel independently invent precedence.

## 3. Build the mathematical stage graph

Before choosing Ascend primitives, express the operator as a small dependency graph:

- which values are read;
- which statistics/reductions are required;
- which intermediates are produced;
- which state persists across steps/chunks;
- which outputs depend on which stages.

Then decide what should be fused or separated. Hardware convenience must not silently change mask semantics, reduction domain, recurrence order or floating-point contract.

This is also where Codex should classify the operator as one or more broad archetypes such as elementwise/broadcast, reduction, scan/recurrent, sparse/gather, matmul/Cube, normalization, attention or composite. The taxonomy helps choose questions; it is not a mandatory implementation template.

## 4. Separate dependency axes from parallel axes

For every logical axis, mark it as one of:

- independent;
- reduction-coupled;
- recurrence/state-coupled;
- producer-consumer coupled.

Choose the logical task ownership before micro-tiling. A task might own a row, head, output tile, sparse block, or an entire chunk/state chain.

Two recurring rules are useful:

- keep a true recurrence/state chain local when practical, and parallelize orthogonal axes around it;
- do not leave an independent axis serial merely because the first baseline was written that way.

Core count follows from legal decomposition and workload size. It is not an objective by itself.

## 5. Reason from physical layout

Logical shape is not enough. Record which dimensions are physically contiguous and what strides the public interface permits.

Prefer a design in which the hot copy direction matches contiguous storage. Host `transpose`, `reshape` or `view` is cheap only while it remains a view; `.contiguous()`, pad or dtype conversion can materialize a full tensor.

Do not automatically move every host transform into the kernel either. Decide based on legality, target API support, tail behavior and total data movement.

## 6. Put runtime scheduling in Host Tiling

Host Tiling should derive internal runtime decisions from shape/dtype/mode/SOC evidence, for example:

- block/core count;
- tile sizes and loop counts;
- TilingKey or implementation regime;
- workspace offsets/sizes;
- aligned/full-tile versus tail path;
- mode-specific internal flags.

Keep the number of regimes small. A new regime should correspond to a real algorithm/resource/hardware-path difference, not a public testcase ID.

Every regime boundary needs correctness cases immediately below, at and above the boundary.

## 7. Design full tiles and tails separately

Make the common aligned/full-tile path explicit and simple. Put partial-tile handling behind the legal padding, mask or tail-copy mechanism.

Do not make every tile use a slow tail path solely to simplify code, and never let a fast path read/write outside the contract.

A generic boundary matrix should include the smallest legal size plus values around relevant hardware alignment boundaries.

## 8. Plan memory from lifetimes

Before allocating buffers, list simultaneous live values and their ownership:

- register/local scalar state;
- UB temporaries;
- L1/L0 operands/accumulators;
- workspace exchange/state;
- GM inputs/outputs.

Separate long-lived state, cross-stage producer/consumer exchange and short scratch. Their reuse distances and required ring depths are different.

Two variable names do not require two physical buffers, and two uses of one buffer are not safe to alias if their lifetimes overlap or asynchronous pipelines still own the data.

Recompute the budget whenever tiling, fusion, dtype or buffering changes.

## 9. Define the numerical contract explicitly

For each stage record:

`storage dtype -> cast -> compute dtype -> accumulator dtype -> output cast`

Also record any required accumulation order or comparison semantics.

For FP16/BF16 work, FP32 accumulation is common but not automatic: follow the contract/reference and target primitive behavior. Mathematical equivalence does not imply bitwise floating-point equivalence. Signed zero, cast rounding, saturation, tree-vs-linear reduction order and padding identities can all matter.

Padding values must be operation-specific. Zero is not a universal neutral value for max/min/softmax/compare paths.

## 10. Operator-family architecture questions

These are prompts, not required implementations.

### Elementwise / broadcast

Choose the independent tile axis and broadcast staging first. Establish a simple `GM -> UB -> Vector -> GM` correctness skeleton before adding queues or deeper pipelining.

### Reduction / normalization

Decide who owns a complete reduction. If a core owns only a partial reduction, include merge/write traffic and numerical-order changes in the design. Count how many complete GM passes the baseline algorithm requires before considering online/pass-fused variants.

### Scan / recurrent

State initialization, carried-state ownership and finalization are semantic decisions. Prefer keeping the dependency chain with one owner and vectorizing/parallelizing independent orthogonal work.

### Sparse / gather / paged

First define index/page/chunk semantics and output order. Only then decide whether adjacent fragments can be coalesced or staged directly into UB/L1. Do not reorder semantically ordered sparse entries for locality unless the contract permits it.

### Cube / matmul

Choose M/N/K task ownership and identify whether one operand has materially higher reuse. Resident and streaming roles can be asymmetric. Fit L1/L0 capacity and preserve useful multicore parallelism before adding ping-pong/preload.

### Mixed Cube + Vector

Define the producer-consumer protocol before coding: producer, consumer, intermediate location, true-ready edge, reuse/overwrite edge and maximum safe in-flight distance. Flag type, ring depth and stage count come after this dependency model, not before it.

## 11. Specialization needs a generic fallback

Use specialization/TilingKey when dtype, shape regime, small fixed factor or hardware path creates a stable implementation difference. Keep a contract-complete fallback unless the contract truly excludes the other cases.

Do not create a new template because one public benchmark shape is important.

## 12. Keep local adaptation separate from the platform source

When local hardware differs from the official target, keep target CANN/SOC source authoritative. A temporary local adaptation should change platform plumbing only when possible.

Do not let the local validation mirror become a second algorithm, and do not present local proxy results as target-platform proof.

## 13. Build the correctness matrix before performance work

The baseline matrix should derive from semantics and hardware boundaries:

- every legal mode;
- every required dtype;
- optional-input present/absent forms;
- smallest legal shapes;
- alignment-1 / alignment / alignment+1;
- tail cases;
- public representative cases;
- independent stress/generalization cases;
- precision-sensitive values when relevant.

On failure, first localize it to contract, tiling, layout, tail, precision, state, synchronization or target API. Add the smallest regression case before redesigning multiple mechanisms at once.

Only after the correctness baseline is stable should the task enter `agent_loop.py diagnose` and the performance pattern library.

## Codex autonomy contract

The Harness intentionally does **not** decide the following automatically:

- exact mathematical reformulation when several are legal;
- tile sizes, block count or core count;
- queue/stage/ring depth;
- exact UB/L1/L0 layout;
- number or thresholds of TilingKey regimes;
- whether a candidate fusion/specialization is worth its complexity.

Codex should make these decisions from the task contract, official Ascend skill guidance, target build/runtime evidence and the current source. It may reject a Harness suggestion, but it should record why.

The Harness is successful when it exposes missing questions early and narrows unsafe choices, not when every operator looks structurally identical.

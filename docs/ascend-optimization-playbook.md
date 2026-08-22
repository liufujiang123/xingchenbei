# Ascend-first kernel optimization playbook

This playbook turns Ascend hardware characteristics into an explicit optimization phase for the competition harness. It complements the official `ascendc-operator-performance-eval` and `ascendc-operator-performance-optim` skills; it does not replace them.

## Why this exists

A generic loop such as `profile -> try optimization -> benchmark` is not enough. Ascend AI Core exposes independent instruction pipelines/queues (for example Vector, Cube, MTE and Scalar), several on-chip memory levels, and explicit synchronization. A useful agent should reason about those resources before generating a candidate.

The goal is:

`operator contract -> operator class -> resource/dataflow model -> measured bottleneck -> matching Ascend patterns -> one candidate -> gates -> keep/reject`

## 1. Classify the operator before optimizing

Every performance pass must classify the hot path as one of:

- `vector`: primarily SIMD/Vector work. Typical dataflow: `GM -> UB -> Vector -> UB -> GM`.
- `cube`: primarily matrix/Cube work. Typical dataflow: `GM -> L1 -> L0A/L0B -> Cube -> L0C -> FixPipe -> ...`.
- `mixed_cv`: Cube produces/consumes tiles together with substantial Vector post/pre-processing.

Do not propose Cube/Vector parallelism for a pure Vector operator. For a Vector operator, the analogous pipeline opportunity is usually MTE2/MTE3 overlap with V plus multicore/UB improvements.

## 2. Write the resource graph

Before code changes, record which resources each stage uses and the true dependencies between stages.

### Vector

Typical queues/resources:

- Scalar/control (`S`)
- GM -> UB transfer (`MTE2`)
- Vector compute (`V`)
- UB -> GM transfer (`MTE3`)
- UB capacity and queue buffers

Ask whether the implementation is effectively:

`CopyIn(n) -> Compute(n) -> CopyOut(n) -> CopyIn(n+1)`

when independent queues could instead overlap work on different tiles:

`CopyIn(n+1) || Compute(n) || CopyOut(n-1)`.

### Cube

Typical resources include MTE2, MTE1, Cube (`M`), FixPipe, L1, L0A/L0B/L0C. Look for data-loading or FixPipe gaps around Cube work and for poor matrix-tile utilization.

### Mixed Cube + Vector

Treat Cube and Vector as producer/consumer stages, not automatically as a serial `C then V` sequence. For independent tiles a desired steady state often resembles:

`Cube(tile n+1) || Vector(tile n)`

with synchronization only at true data dependencies. Workspace/buffer reuse can create false dependencies: a producer may wait only because the consumer has not released the same slot. Evaluate a deeper workspace ring when profile evidence shows both C and V idle gaps.

## 3. Pipeline techniques are first-class candidates

### Double buffer / ping-pong

Different AI Core instruction queues can execute independently. When multiple independent tiles exist and on-chip capacity permits, double buffering can hide movement behind compute. This is not a universal win: extra buffers reduce available UB/L1 and may hurt small shapes.

A candidate must therefore state:

- which stages overlap;
- which two (or more) tiles are live at once;
- buffer ownership/lifetime;
- required event/barrier edges;
- extra UB/L1/workspace bytes;
- expected bottleneck hidden by the overlap.

### C/V overlap

For `mixed_cv`, inspect whether Cube and Vector have useful independent work on adjacent tiles. Candidate families include:

- C(tile n+1) overlapping V(tile n);
- avoiding unnecessary full-stage barriers;
- separate workspace slots/ring buffers to eliminate false producer-consumer serialization;
- measured AIC/AIV work-ratio tuning when one side is consistently bound.

The official Ascend GroupedMatmul optimization case is an example of this method: it analyzes Cube/Vector gaps, changes AIC/AIV balance, deepens workspace buffering to reduce C/V waiting, and then adds Vector double buffering. Treat such numbers as an example, not a portable constant for other operators/SOCs.

### MTE/V overlap

For `vector`, inspect whether CopyIn/CopyOut and Vector compute are serialized. Use TPipe/TQue/ping-pong only when there are enough independent tiles and the UB budget still supports efficient vector work.

## 4. Other Ascend optimization dimensions

After architecture/pipeline classification, consider only the dimensions supported by evidence:

- **multicore**: task granularity, block/core count, shape-aware tiling, tail balance;
- **data movement**: contiguous/coalesced transfers, aligned full-tile fast paths, DataCopyPad only where needed;
- **memory hierarchy**: UB/L1/L0 reuse, avoid GM materialization of intermediates, balance reuse against occupancy;
- **API usage**: avoid redundant Duplicate/cast/copy, use efficient supported Ascend C APIs for the target CANN/SOC;
- **scalar/control**: hoist invariants, avoid repeated integer div/mod in hot loops, specialize only stable/common small modes;
- **precision-aware reduction**: FP32 accumulation where required; any changed reduction order is a precision-risk candidate;
- **internal fusion**: keep compatible pre/post-processing close to the producer instead of round-tripping through GM/workspace when the competition interface permits.

## 5. Profiling evidence expected by the agent

Do not infer pipeline utilization only from source shape. When profiling is available, capture evidence such as:

- end-to-end kernel latency after warmup;
- per-case median and variance;
- core/block utilization and work distribution;
- instruction/cycle utilization (Vector/Cube/MTE/Scalar where exposed);
- pipeline timeline gaps and producer/consumer waits;
- bytes or repeated transfers implied by the implementation;
- UB/L1/L0/workspace budget for the candidate.

Profile output should answer a question, for example: “Is Vector waiting on MTE2?” or “Are C and V serialized by workspace reuse?”

## 6. Candidate generation

After classification and bottleneck identification, use the machine-readable registry:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class vector \
  --bottleneck pipeline \
  --bottleneck memory
```

For a mixed Cube/Vector operator:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class mixed_cv \
  --bottleneck pipeline
```

The output is a candidate shortlist, not permission to apply every pattern.

## 7. One-candidate rule

A candidate should change one main mechanism so its contribution can be measured. Examples:

- only change queue depth / double buffering;
- only change workspace ring depth;
- only change AIC/AIV work partition;
- only change tile size/core utilization;
- only replace aligned DataCopyPad calls with DataCopy fast paths.

Do not combine C/V overlap, new tiling and precision-changing reduction in the same experiment.

## 8. Promotion gates

Every candidate still passes the repository gates:

1. public interface unchanged;
2. target-platform build passes;
3. configured correctness matrix passes;
4. same-case benchmark improves beyond noise;
5. no required shape/dtype/mode domain is narrowed;
6. platform-specific conclusions are labeled correctly (for example A3 proxy is not 910B proof).

Record failed candidates too.

## MhcExpand classification note

MhcExpand currently belongs to the `vector` family: forward is data replication and backward is a small-factor reduction. It has no meaningful Cube stage, so C/V parallelism should not be forced into this operator. Its Ascend-first search should emphasize:

- MTE2/V/MTE3 overlap and double buffering where enough tiles exist;
- multicore utilization for low-S/high-D and high-S/low-D shapes;
- aligned copy fast paths;
- UB source/accumulator reuse;
- scalar task-mapping overhead;
- precision-safe reduction specialization.

This distinction is exactly why the framework classifies the operator before selecting techniques.

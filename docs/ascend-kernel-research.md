# Ascend kernel research notes

This is a compact provenance note for `xingchen-kernel-optimizer`. It records mechanisms distilled from public high-performance Ascend/NPU repositories so the runtime skill can stay small.

The rule for admitting a pattern is:

1. it appears in a real high-performance implementation or optimization guide;
2. there is a plausible Ascend hardware/dataflow reason for the gain;
3. it can be expressed as an applicability signal + experiment + risk gate;
4. fixed parameters from one kernel are not promoted to global rules.

No third-party implementation is copied into the competition kernels by this document.

## Repositories reviewed

### flashserve/flash-linear-attention-npu

Representative files reviewed:

- `fla/ops/ascendc/common/kernel_utils/block/block_mmad_pingpong_tla.hpp`
- `fla/ops/ascendc/common/kernel_utils/vector/regbase.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_common.h`
- `chunk_gated_delta_rule_bwd_dhu_design.md`

Distilled points:

- multi-level L1/L0/UB stage counts are explicit resources constrained by memory and event IDs;
- L1 residency includes reuse detection/state restoration, not merely "allocate L1 and keep it";
- a mixed C/V kernel can use a **credit window** so Cube may lead Vector by a bounded number of tasks instead of lockstep ready/wait;
- the safe credit window follows the true dependency distance and workspace capacity;
- long-lived and short-lived temporaries can require different ring depths;
- over-sized workspace rings can lose cache locality and increase MTE/FixPipe misses;
- recurrent chunk state is kept on one task/core while independent sequence/head axes are parallelized;
- MicroAPI `RegTensor` kernels are used for low-level Vector microkernels when the workload and architecture justify the complexity.

Most surprising reusable lesson: deeper buffering is not monotonic. Buffer depth is a scheduling parameter **and** a working-set/cache parameter.

## MinghuasLab/flash-attention-npu

Representative files reviewed:

- `csrc/ascend910/flash_attn_npu/qk_matmul.hpp`
- `csrc/ascend910/flash_attn_npu/CombineScale.hpp`

Distilled points:

- keep reused operands such as Q resident while streaming the opposite operand;
- paged/fragmented KV can be assembled directly into L1 for the matrix operation rather than gathered into a full GM intermediate;
- UB regions can be reused/aliased according to lifetime after earlier values are dead;
- tile choice must account for physical layout/stride and contiguous movement, not only theoretical capacity.

Most surprising reusable lesson: sparse/paged gather and layout repair are often best treated as part of the **staging path**, not as a separate materialization kernel.

## tile-ai/tilelang-ascend

Representative material reviewed:

- `examples/flash_attention/fa_opt/flash_attention_performance_optimization_zh.md`
- `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`
- `.agents/skills/tilelang-perf-optimization/SKILL.md`
- `.agents/skills/tilelang-perf-optimization/references/optimization-guide.md`
- `.agents/skills/tilelang-perf-optimization/references/vector-practices/vector_reduce_pass_fusion.md`

Distilled points:

- optimize toward a dominant/bound pipeline by hiding shorter MTE/Vector/Cube work underneath it;
- explicitly model pipeline prologue/main/epilogue rather than only the steady-state loop;
- cross-core synchronization can be **batched** over several tiles with a final tail signal;
- only buffers accessed asynchronously by MTE producers/consumers need double buffering; double-buffering pure Vector temporaries wastes UB;
- long serial scans can batch several scan steps into UB, keep recurrent state local, and vectorize an orthogonal independent axis;
- repeated reduction passes over GM can sometimes be reduced with online correction formulas;
- scalar slice loops should become tile-wide Vector operations when semantics allow;
- generated target code should be inspected when a dtype/shape unexpectedly falls onto a scalar or inferior hardware path;
- optimization ordering matters because layout, live-buffer count and memory planning interact.

Most surprising reusable lesson: "double buffer the kernel" is too coarse. The correct question is **which individual buffer has an asynchronous lifetime hazard**.

## ascend-catlass/catlass

Representative material reviewed:

- `docs/zh/2_Design/01_kernel_design/04_matmul_summary.md`
- `tools/tuner/README.md`

Distilled points:

- ordinary ping-pong can still leave MTE2 bubbles between output blocks; **preload across block boundaries** addresses startup/transition gaps;
- a resident operand should not automatically use the same multi-buffering policy as the streaming operand;
- when many cores start the same K traversal at the same phase, **ShuffleK-style phase shifting** can reduce simultaneous reads to the same GM region;
- task/swizzle order is a cache/bandwidth decision, not merely a load-balance decision;
- shape regimes may need different copy instructions or kernel templates; a generic conversion path can be poor for small dimensions;
- Split-K trades added parallelism against extra writes/reduction, so it is a modelled tradeoff rather than an automatic optimization;
- padding/reformat can be profitable when it buys much better DMA bandwidth, but transform cost must be included;
- small kernels may be scalar-control bound enough to justify a simplified scheduler;
- CATLASS tuner searches tile shapes/layouts/swizzles with hardware constraints and on-board measurements rather than blind enumeration.

Most surprising reusable lesson: task traversal order can be optimized for **bandwidth deconfliction**, even when all tasks access the same total bytes.

## Ascend/triton-ascend

Representative material reviewed:

- `docs/en/autotune_guide.md`
- `python/tutorials/03-matrix-multiplication.py`

Distilled points:

- candidate tilings are generated and filtered by on-chip memory, alignment and core-utilization constraints before benchmarking;
- runtime shape/configuration attributes form the autotune cache key;
- tiling can be tuned jointly with other parameters such as multibuffering;
- profiler-based on-chip timing is useful for very short kernels;
- grouped program ordering is a separate optimization dimension for cache reuse;
- assumptions/invariants can help remove address-calculation overhead in generated code.

Most useful framework lesson: use a **small hardware-pruned search space keyed by shape regime**, not one globally fixed tiling and not an unbounded brute-force search.

## Patterns admitted to the registry

The v2 registry keeps a deliberately small set of mechanisms:

- multicore balance;
- dependency-aware partition;
- working-set/liveness-aware buffering;
- cache/locality and bandwidth-deconflicting schedule;
- direct on-chip staging for fragmented data;
- scalar/control reduction;
- bounded shape-regime autotuning;
- target hardware-path audit;
- Vector MTE/V pipeline;
- Vector reduce-pass fusion;
- blocked scan + orthogonal vectorization;
- Vector instruction fusion;
- advanced MicroAPI register microkernel;
- Cube cross-block preload;
- asymmetric operand residency;
- Cube reduction-axis split for occupancy;
- mixed C/V credit-window pipeline;
- mixed C/V synchronization batching;
- mixed C/V liveness-sized workspace rings;
- advanced internal C/V handoff/fusion.

`tools/ascend_perf_plan.py` hides advanced patterns by default.

## Ideas intentionally not imported as global rules

The following may be valid in their source kernels, but are **not** encoded as universal constants:

- a particular `num_stages`;
- a particular C/V synchronization interval;
- a particular credit count or ring depth;
- a fixed AIC:AIV ratio;
- a fixed L1/L0/UB tile shape;
- source-project-specific transfer-size thresholds;
- source-project-specific alignment numbers unless the target API for the current task proves the same requirement;
- TileLang pass configuration rules as if they were native AscendC rules;
- A2/A3/950-specific MicroAPI or instruction behavior without target compatibility evidence;
- CUDA/A100 performance numbers appearing in shared Triton tutorial material.

The agent should copy the **reasoning test**, not the parameter.

## How Codex should use this research

Normal path:

```text
classify operator
-> identify real dependency and live ranges
-> measure/tag bottleneck
-> run ascend_perf_plan.py
-> choose one core pattern
-> implement one candidate
-> target build + correctness + same-case measurement
-> keep/reject
```

Advanced path is allowed only after the normal path has evidence that a lower-level target-specific mechanism is justified.

For every promoted change, the optimization log should be able to answer:

- what hardware/resource symptom existed?
- why this pattern should change it?
- what extra memory/synchronization/precision risk it introduces?
- did the measured result match that hypothesis?

That keeps the repository useful to Codex without turning it into a copy of the source projects.

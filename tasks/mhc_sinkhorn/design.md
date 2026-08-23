# MhcSinkhorn retained optimized design

Status: **implemented and locally validated on 2026-08-22 UTC**.

This design is derived from the frozen platform execution ABI and the
non-conflicting mathematical sequence retained in `TASK.md`. It is an
independent implementation; no historical kernel queue, buffer, layout,
pipeline, blockDim, or task-ownership choice is treated as a baseline.

## 1. Immutable public contract and required domain

- Op name: `MhcSinkhorn`.
- Inputs: required `logits`, optional `mask`, in that order.
- Output: required `weights` with the same dtype and shape as `logits`.
- Attributes: optional `iterations` (default `20`) and `eps` (default `1e-6`),
  in that order.
- Storage dtype: FP16 or FP32; format ND.
- Shape: `[...,N,N]`, rank at least two, `N in {4,6,8}`. Every product of
  leading dimensions is supported; it is not narrowed to visible cases.
- Mask regimes are absent, one element, or the same element count as logits.
  A present mask must have the logits dtype.
- The public kernel arguments remain
  `logits, mask, weights, workspace, tiling`. No intermediate output is added.
- `iterations` follows the non-conflicting statement range `[1,100]`. No
  legal range for `eps` is established by the frozen ABI or statement, so the
  Host does not invent one; it passes the float through to the formula.

## 2. Mathematical stage and dependency graph

For each independent matrix, the retained FP32 sequence is:

```text
load logits -> FP32 -> add mask before softmax
    -> stable row softmax
    -> add eps to every softmax result
    -> column denominator = FP32 column sum + eps
    -> column division
    -> repeat iterations-1 times:
           row denominator = FP32 row sum + eps
           row division
           column denominator = FP32 column sum + eps
           column division
    -> cast to storage dtype -> weights
```

The stable softmax denominator itself has no `eps`; `eps` is added to the
normalized softmax result. This is intentionally distinct from adding eps to
logits or the softmax denominator.

## 3. Independent, reduction, and recurrence axes

| Axis | Classification | Consequence |
|---|---|---|
| flattened leading dimensions | independent | matrices may be assigned to different AIV cores |
| row/column length `N` | reduction | FP32 reductions remain inside one core |
| Sinkhorn iteration | recurrence | never split across cores or reordered |

One core owns a whole matrix from initial load through final store. There is no
cross-core partial reduction, workspace exchange, or synchronization.

## 4. Logical task/core ownership

`matrixCount = product(logits.shape[:-2])`. Host chooses
`usedCoreNum = min(matrixCount, availableAivCores)` for nonempty workloads.
Core `c` receives a balanced contiguous interval using quotient/remainder
partitioning. For an empty leading domain Host launches one core and that core
executes zero matrix loops, preserving the full shape domain without a zero
blockDim launch.

## 5. Physical layout, contiguity, and tails

Inputs and output use compact ND storage. Each matrix is a contiguous `N*N`
segment; full masks use identical flattened element offsets, while a scalar
mask always reads element zero.

GM/UB copies use byte-counted `DataCopyPad`, so the unaligned `N=6` matrix
sizes are legal:

| N | elements | FP16 GM bytes | FP32 GM bytes |
|---:|---:|---:|---:|
| 4 | 16 | 32 | 64 |
| 6 | 36 | 72 | 144 |
| 8 | 64 | 128 | 256 |

`N=8` uses compact 64-element matrices and batches up to 32 per core-local
tile. `N=4/6` batches up to four matrices and stores each logical row in an
8-float physical row. Its padding is initialized to zero; max reductions still
use logical `N`, while validated sum reductions may include only zero padding.
Padded elements are never written back to GM.

## 6. Host Tiling responsibilities and regime boundaries

Host validates required tensors, rank, square trailing dimensions, N, dtype,
mask dtype/element count, and iteration range. It derives:

```text
DT_LOGITS in {FP16, FP32}
N         in {4, 6, 8}
MASK_MODE in {0, 1, 2}
```

TilingData contains only runtime work needed by every specialization:
`matrixCount`, `matrixSize`, `usedCoreNum`, `iterations`, and `eps`.
Workspace is zero bytes. InferShape copies the logits shape; InferDataType
copies its dtype.

The three template dimensions are algorithm/resource boundaries, not testcase
IDs. The source remains AIV-only and exposes one public ABI.

## 7. Full-tile and tail strategy

The kernel-local tile is up to 32 complete matrices for `N=8`, or four for
`N=4/6`. `N` is compile-time fixed. A core processes its quotient/remainder
interval in batches and handles the final partial batch with `batchCount`;
matrix ownership and every matrix's complete iteration chain remain local.

## 8. UB, queue, buffer, and lifetime plan

There is no GM iteration spill and no cross-core pipeline. Matrix batching is
used to increase Vector work density, while each matrix's recurrence remains
serial.

| Resource | Lifetime | Purpose |
|---|---|---|
| one-depth VECIN queue | one matrix batch load | logits, then optional mask, reusing one allocation |
| FP32 state TBuf | core lifetime | persistent batched matrices through all iterations |
| FP32 row-stats TBuf | core lifetime | row/column reduction results |
| FP32 row-broadcast TBuf | core lifetime | reduction broadcast and N=8 pairwise tree temporary |
| conditional FP32 mask TBuf | core lifetime, FP16 full-mask specialization only | cast full FP16 mask before add |
| one-depth VECOUT queue | one matrix batch store | output cast/copy |
| workspace | none | no cross-core or spill state |

The maximum explicit UB allocation is 33,792 bytes: input/output queues,
8,192-byte FP32 state, 1,024-byte row statistics, 8,192-byte broadcast/tree
storage, and the conditional mask-cast buffer. This fits without workspace.

## 9. Precision contract

| Stage | Precision/order |
|---|---|
| logits/mask storage | FP16 or FP32 |
| mask addition | FP32 after independent casts |
| max, exp, sums, reciprocal, division, recurrent state | FP32 |
| reductions | validated BlockReduce row reductions; N=8 pairwise column tree; N=4/6 fixed logical/padded reductions |
| output | one final cast to logits dtype |

The promoted reduction order differs from the original scalar reference and
was therefore treated as an isolated numerical reformulation. FP16/FP32 full
correctness passed before it was retained. Further order changes require the
same treatment.

## 10. Correctness matrix and evidence

The independent NumPy reference covers both dtypes, all N values, ranks two,
three, and four, all mask regimes, default/nondefault iterations and eps,
finite extreme logits, single/multiple matrices, empty leading domains, and
invalid dtype/shape/mask/iteration contracts.

Executed evidence on 2026-08-22 UTC:

- CANN 8.5.0 910B official target: Host, FP16 kernel, FP32 kernel, generated
  ACLNN library, and `.run` package built successfully.
- Local SOC was detected through ACL as `Ascend910_9382`.
- A temporary mirror changed only CMake and OpDef target strings to
  `ascend910_93`; the official source remained `ascend910b`.
- CPU contract/reference suite: 86 passed.
- Local A3 ACLNN comparison: 21 passed, covering
  `2 dtypes * 3 N values * 3 mask modes` plus three nondefault/extreme cases.

## Analyzer decisions

The full design-analyzer catalog was reviewed. Three patterns were selected:

- `semantics.stage_graph`: used to freeze the mask/softmax/eps/normalization
  stage ordering before hardware choices.
- `parallelism.axes_and_ownership`: used to keep the recurrent iteration and
  reductions within one core while parallelizing only matrices.
- `precision.compute_contract`: used to make FP32 state and reduction
  behavior explicit.

Machine hints were treated only as navigation. No static signal is reported as
measured performance evidence.

## Retained optimized architecture and remaining limits

The initial scalar reference established the following model:

- event timing shows a large fixed cost only below roughly one natural
  48-core wave; after that, per-matrix throughput is nearly flat;
- the iterations fit has R-squared 0.999999 and only about 4.75% fitted fixed
  cost at iterations=20;
- msprof reports a 99.4% median Scalar-pipe ratio and only 0.1% Vector-pipe
  ratio for the representative steady-state case.

The retained optimized kernel replaces that scalar recurrent body with
compile-time N specializations, matrix batches, Vector BlockReduce/Brcb paths,
and a pairwise N=8 column tree. The local orthogonal proxy score improved from
3986.280861 us to 198.175802 us (about 20.1x). At iterations=100, the retained
N=8 profile reports roughly 78% Vector active but only about 20.8% FP32 plus
miscellaneous arithmetic utilization. MTE and UB conflict are small. The
remaining hypothesis is short reduction/division dependency issue gaps, not a
GM-residency failure; the profiler exposes no direct stall-reason field, so
this remains a static hypothesis rather than a measured stall attribution.

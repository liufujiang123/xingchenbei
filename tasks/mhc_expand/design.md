# MhcExpand specification-phase design

## Evidence status

- **PLATFORM FACT** — The live CANNJudge statement snapshot is [problem.md](problem.md); verified contract details and provenance are in [TASK.md](TASK.md).
- **PLATFORM FACT** — The live package downloaded on 2026-08-18 is authoritative over the stale checked-in package where they differ.
- **TEMPLATE FACT** — No file under `workspace/code/` has been modified in this phase.
- **TEMPLATE FACT** — Repository guidance routes architecture/tiling work through `ascendc-operator-design`; its design template, reduction tiling guidance, general tiling principles, and the repository-local `xingchen-kernel-optimizer` workflow were inspected.
- **DESIGN PROPOSAL** — Generic skill examples are advisory only; all proposed choices below remain subordinate to the verified CANNJudge contract and must be revisited after the unresolved direction-selector contract is confirmed.

## Mathematical semantics

### Forward expansion

- **PLATFORM FACT** — Input: `x in {BF16,FP16}^{S x D}`, with `S>0`, `D>0`.
- **PLATFORM FACT** — Attribute: `mhc_mult=m`, a positive integer; current package default `m=2`.
- **PLATFORM FACT** — Output: `o in {BF16,FP16}^{S x m x D}`, with `dtype(o)=dtype(x)`.
- **PLATFORM FACT** — Equation: `o[i,k,j]=x[i,j]` for `0<=i<S`, `0<=k<m`, `0<=j<D`.
- **PLATFORM FACT** — The output is materialized storage, equivalent to `x.unsqueeze(1).expand(-1,m,-1).clone()`.

### Backward reduction

- **PLATFORM FACT** — Logical input: `o_grad in {BF16,FP16}^{S x m x D}`.
- **PLATFORM FACT** — Logical output: `x_grad in {BF16,FP16}^{S x D}`, with the same dtype as `o_grad`.
- **PLATFORM FACT** — Equation: `x_grad[i,j]=sum(k=0..m-1,o_grad[i,k,j])`.
- **PLATFORM FACT** — The live package represents both logical signatures with physical ports `x` and `o`, plus optional `mhc_mult` and `backward` attributes.
- **UNRESOLVED** — The public platform evidence does not explicitly say whether `backward=true` selects reduction and `backward=false` selects expansion, nor what omission of `backward` means.

## Interface reconciliation

| Evidence | Input | Output | Attributes | Dtypes | Classification |
|---|---|---|---|---|---|
| Checked-in `workspace/code/` | `x` | `y` | required `expand_num:int` | FP16, FP32 | **TEMPLATE FACT** |
| Fresh CANNJudge package | `x` | `o` | optional `mhc_mult:int=2`; optional `backward:bool` | BF16, FP16 | **PLATFORM FACT** |
| Platform statement, forward logical names | `x` | `o` | `mhc_mult` | BF16, FP16 | **PLATFORM FACT** |
| Platform statement, backward logical names | `o_grad` | `x_grad` | `mhc_mult` | BF16, FP16 | **PLATFORM FACT** |

- **TEMPLATE FACT** — The checked-in OpDef is not the current submission interface.
- **PLATFORM FACT** — The fresh package is explicit platform proof of the changed submission interface.
- **UNRESOLVED** — The unrelated softmax footer in the live statement is a platform-side contradiction and supplies no usable MhcExpand interface information.

## Current template architecture

- **TEMPLATE FACT** — Root CMake finds the ASC package, selects `ascend910b`, creates package `custom`, and adds `op_host` and `op_kernel`.
- **TEMPLATE FACT** — Host CMake generates operator code, builds `cust_optiling`, generates ACLNN source, builds `cust_opapi`, and packages both libraries.
- **TEMPLATE FACT** — Kernel CMake builds and packages `ascendc_kernels` linked to `cust_optiling`.
- **TEMPLATE FACT** — Checked-in Host Tiling reads input dtype, element count, byte size, and `expand_num`; it selects a dtype tiling key, writes only `length`, launches all available AIV cores, and requests zero workspace.
- **TEMPLATE FACT** — Fresh Host Tiling has the same skeleton but reads `mhc_mult` and `backward`; neither value is stored in TilingData.
- **TEMPLATE FACT** — Both InferShape and InferDataType return success without setting output metadata.
- **TEMPLATE FACT** — Both TilingData versions contain only `uint32_t length`.
- **TEMPLATE FACT** — Both kernels register the TilingData, instantiate a dtype-specialized class, and call empty `Init` and `Process` methods.
- **TEMPLATE FACT** — Fresh tiling-key selection supports FP16 and BF16; checked-in selection supports FP32 and FP16.

## Required InferShape behavior

- **PLATFORM FACT** — Forward mode requires rank-2 input `[S,D]` and output `[S,mhc_mult,D]`.
- **PLATFORM FACT** — Backward mode requires rank-3 input `[S,mhc_mult,D]` and output `[S,D]`.
- **DESIGN PROPOSAL** — After the mode mapping is confirmed, InferShape should read `mhc_mult` and the direction attribute, validate the legal rank and positive dimensions, validate that backward input dimension 1 agrees with `mhc_mult`, and set the exact output shape above.
- **DESIGN PROPOSAL** — Shape products and offsets should use 64-bit arithmetic and explicitly guard overflow before narrowing any launch/tiling field.
- **UNRESOLVED** — Required behavior for invalid rank, non-positive dimensions, non-positive `mhc_mult`, or backward shape mismatch is not published.

## Required InferDataType behavior

- **PLATFORM FACT** — Output dtype must equal input dtype in both directions.
- **PLATFORM FACT** — The live legal dtype set is exactly `{DT_BF16, DT_FLOAT16}`.
- **DESIGN PROPOSAL** — InferDataType should reject any dtype outside that set and set output 0 to input 0's dtype.

## Proposed simplest correctness baseline

### Mode-independent Host Tiling

- **DESIGN PROPOSAL** — Base the future implementation on the fresh package's `x -> o`, `mhc_mult`, `backward`, BF16/FP16 contract, but do not edit code until the direction-selector ambiguity is resolved.
- **DESIGN PROPOSAL** — Pass at least `S`, `D`, `mhcMult`, `direction`, `taskCount`, `tileD`, `usedCoreNum`, and tail-task information in TilingData; `length` alone cannot express the operation.
- **DESIGN PROPOSAL** — Define one independent task as a contiguous hidden-dimension slice of one token row: `(s, dBegin:dEnd)`. Use `taskCount=S*ceil_div(D,tileD)` and distribute tasks across `min(AIV core count, taskCount)` cores.
- **DESIGN PROPOSAL** — Derive `tileD` from the runtime UB size and dtype size, round normal tiles to a 32-byte multiple, and carry the logical tail length separately so non-aligned `D` never causes an out-of-bounds GM access.
- **DESIGN PROPOSAL** — Request zero user workspace for the baseline because tasks are independent and all temporary state fits in per-core UB.

### Forward data path

- **DESIGN PROPOSAL** — For each `(s,d-tile)` task, copy the contiguous `x[s,dBegin:dEnd]` tile from GM to one UB buffer once, then write that same UB tile to `o[s,k,dBegin:dEnd]` for every `k in [0,m)`.
- **DESIGN PROPOSAL** — This row/tile mapping preserves coalesced contiguous copies, reuses each input load `m` times, and avoids division/modulo per output element.
- **DESIGN PROPOSAL** — The correctness-first UB plan needs one input tile of `tileD*2` bytes plus alignment/padding; a distinct output buffer is not mathematically required if the supported LocalTensor-to-GM copy path can reuse the loaded tile.
- **DESIGN PROPOSAL** — Forward performs no arithmetic and no dtype conversion; copying bit patterns preserves NaNs, infinities, signed zero, and all finite BF16/FP16 values exactly.

### Backward data path

- **DESIGN PROPOSAL** — For each `(s,d-tile)` task, initialize an FP32 accumulator tile to zero, loop over `k`, copy the contiguous `x[s,k,dBegin:dEnd]` gradient tile into UB, cast it to FP32, accumulate, cast once to the input dtype, and write `o[s,dBegin:dEnd]`.
- **DESIGN PROPOSAL** — FP32 accumulation followed by one final cast matches the referenced TileKernels strategy and is the safest baseline for both BF16 and FP16.
- **DESIGN PROPOSAL** — A single-buffer baseline needs symbolic UB storage of `2*tileD` bytes for the input, `4*tileD` for the cast temporary, `4*tileD` for the accumulator, and `2*tileD` for the result, totaling `12*tileD` bytes plus alignment/padding.
- **DESIGN PROPOSAL** — No atomic operation or cross-core synchronization is required because exactly one task owns each output slice.

### Ascend C operation sequence

- **DESIGN PROPOSAL** — Forward sequence: `DataCopyPad/valid-tail load -> repeat m times { valid-tail store }`.
- **DESIGN PROPOSAL** — Backward sequence: `Duplicate(acc,0) -> repeat m times { valid-tail load -> Cast(FP16/BF16 to FP32) -> Add(acc,acc,tmp) } -> Cast(FP32 to FP16/BF16) -> valid-tail store`.
- **DESIGN PROPOSAL** — Exact CANN 8.5 API overloads and queue/buffer mechanics must be confirmed during code generation/compile validation rather than assumed from generic skill snippets.

## Boundary behavior to cover

- **PLATFORM FACT** — Required published boundaries include `S=1`, `D=1`, and non-32-byte-aligned dimensions.
- **DESIGN PROPOSAL** — Exercise `m=1` even though published coverage emphasizes 2/4/8; it is inside the stated positive-integer domain and should reduce to identity forward/backward.
- **DESIGN PROPOSAL** — Exercise small `D` below one data block, tail-only rows, task counts below core count, and very large `S*D*m` products.
- **UNRESOLVED** — Empty tensors and invalid attributes are outside the published legal domain and must not be assigned invented numerical semantics.

## Major correctness risks

- **UNRESOLVED** — Reversing or defaulting the `backward` selector incorrectly would make shape inference and the kernel interpretation mutually incompatible.
- **TEMPLATE FACT** — Implementing against checked-in `y/expand_num/FP32` would violate the current live platform interface.
- **TEMPLATE FACT** — Keeping only `uint32_t length` loses rank, dimensions, expansion factor, and direction and can overflow for large materialized outputs.
- **DESIGN PROPOSAL** — Guard against output-size multiplication overflow before calculating `[S,m,D]` storage or GM offsets.
- **DESIGN PROPOSAL** — Tail copies must never read or write beyond logical tensor bounds; padding in UB must not leak into GM output.
- **DESIGN PROPOSAL** — Backward should not accumulate directly in BF16/FP16 because rounding at every add can amplify error as `m` grows.
- **DESIGN PROPOSAL** — Multicore partitioning must assign each backward output element to exactly one core to avoid races or double counting.
- **UNRESOLVED** — The public statement does not pin an accumulation order or per-case tolerance override, so the baseline should use the most precision-safe reasonable order and rely only on public evaluator feedback.

## Later performance optimization dimensions

- **DESIGN PROPOSAL** — Forward: compare one-load/m-writes UB reuse against output-major contiguous tasking and larger 2D `(token,hidden)` tiles.
- **DESIGN PROPOSAL** — Forward: add double buffering only after the single-buffer baseline passes correctness, measuring whether GM store bandwidth already dominates.
- **DESIGN PROPOSAL** — Backward: tune `tileD`, buffer count, and cast/add pipeline overlap as a function of `m`, `D`, and runtime UB capacity.
- **DESIGN PROPOSAL** — Backward: evaluate vector Add chains, pairwise/tree reduction for larger `m`, and API-supported multi-row copy patterns while preserving FP32 accumulation semantics.
- **DESIGN PROPOSAL** — Both directions: tune core partitioning for small `S`, small/non-aligned `D`, and large `S*ceil(D/tileD)` task counts.
- **DESIGN PROPOSAL** — Both directions: reduce address-arithmetic overhead by precomputing row/tile strides in tiling data without specializing to visible testcase IDs.
- **DESIGN PROPOSAL** — Each meaningful optimization must follow guard -> build -> validate -> benchmark, and only precision-passing candidates may be retained or scored.

## Readiness

- **UNRESOLVED** — **BLOCKED** for correctness implementation.
- **UNRESOLVED** — Concrete missing information: the platform-defined mapping of `backward` values to forward/backward execution and the behavior when optional `backward` is omitted.

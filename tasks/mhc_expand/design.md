# MhcExpand correctness baseline design

## 1. Evidence and design boundary

- **PLATFORM FACT** — The authoritative competition contract and provenance are recorded in [TASK.md](TASK.md); the target environment is CANN `8.5.0` on `ascend910b`.
- **PLATFORM FACT** — The public interface is fixed: required input `x`, required output `o`, optional `mhc_mult` with default `2`, optional `backward`, ND format, and dtypes `{BF16, FP16}`.
- **TEMPLATE FACT** — Before baseline implementation, checked-in `workspace/code/` was byte-for-byte identical to the latest official package. Its visible OpDef is `x -> o`; its dtype TilingKey distinguishes FP16 and BF16.
- **TEMPLATE FACT** — The current workspace keeps that OpDef and dtype TilingKey unchanged, while replacing only the official InferShape, InferDataType, TilingData, Host Tiling, and empty kernel stubs with the baseline described below.
- **DESIGN PROPOSAL** — This document defines the simplest complete correctness baseline. It does not change the public OpDef, does not add user workspace, does not introduce double buffering, and does not specialize for visible testcase IDs.
- **DESIGN PROPOSAL** — The implementation path is a custom Ascend C vector kernel. No matrix/Cube computation is needed.
- **DESIGN PROPOSAL** — `xingchen-kernel-optimizer` gates apply after implementation: establish build and correctness first; benchmark/profile/optimization candidates remain deferred.

## 2. Mathematical contract and physical layout

### 2.1 Forward

- **PLATFORM FACT** — Logical and physical input is `x[S,D]`, with `S>0`, `D>0`.
- **PLATFORM FACT** — Logical and physical output is `o[S,m,D]`, where `m=mhc_mult>0`.
- **PLATFORM FACT** — `o[i,k,j] = x[i,j]` for `0<=i<S`, `0<=k<m`, `0<=j<D`.
- **PLATFORM FACT** — `dtype(o)=dtype(x)` and the output is materialized, not a zero-stride view.

### 2.2 Backward

- **PLATFORM FACT** — Physical input is still named `x`, but logically it is `o_grad[S,m,D]`.
- **PLATFORM FACT** — Physical output is still named `o`, but logically it is `x_grad[S,D]`.
- **PLATFORM FACT** — `o[i,j] = sum(k=0..m-1, x[i,k,j])`.
- **PLATFORM FACT** — ND row-major input offset is `((i*m+k)*D+j)` and output offset is `(i*D+j)`.
- **PLATFORM FACT** — Published coverage includes `m in {2,4,8}`, but the legal-domain wording is every positive integer `m`.

### 2.3 Direction

- **PLATFORM FACT** — Legal rank 2 means forward and legal rank 3 means backward; the two legal signatures are disjoint.
- **TEMPLATE FACT** — `backward` is declared with `.AttrType(OPTIONAL).Bool()` and therefore has no explicit source-level default. A local CANN 9.0 generation diagnostic synthesized `false`; that observation is not a platform fact for CANN 8.5.
- **DESIGN PROPOSAL** — Use the design convention `false=forward`, `true=backward`, but derive the actual mode from rank first.
- **DESIGN PROPOSAL** — If the `backward` pointer is non-null, require it to agree with rank. If it is null, retain the rank-derived mode and never dereference it.
- **UNRESOLVED** — The statement does not literally define the bool mapping or omitted-value behavior. This does not block legal inputs because their rank uniquely determines the operation.

## 3. InferShape design

### 3.1 Validation order

- **DESIGN PROPOSAL** — Read input 0 shape and attributes through null-checked context pointers. Any missing required input/output descriptor is a Host error.
- **DESIGN PROPOSAL** — Accept only rank 2 or rank 3. Rank 2 selects forward; rank 3 selects backward. Any other rank returns a non-success graph status and must not launch a kernel.
- **DESIGN PROPOSAL** — Resolve `mhc_mult` as `*GetInt(0)` when non-null, otherwise use its declared default `2`; require `mhc_mult>0`.
- **DESIGN PROPOSAL** — Read `backward=GetBool(1)` without dereferencing null. If present, reject `rank=2 && backward=true` and `rank=3 && backward=false`.
- **DESIGN PROPOSAL** — Require all published logical dimensions to be positive. Unknown/negative/zero dimensions are outside the published legal domain and return a Host error in this baseline.
- **DESIGN PROPOSAL** — Convert validated positive signed dimensions to `uint64_t` only after validation, and use checked multiplication for `S*D` and `S*D*mhc_mult`.

### 3.2 Forward inference

- **DESIGN PROPOSAL** — For rank-2 input `[S,D]`, set output 0 to `[S,mhc_mult,D]`.
- **DESIGN PROPOSAL** — Validate that `S>0`, `D>0`, `mhc_mult>0`, and that the materialized element count `S*mhc_mult*D` does not overflow the Host/kernel offset domain.

### 3.3 Backward inference

- **DESIGN PROPOSAL** — For rank-3 input `[S,M,D]`, require `S>0`, `M>0`, `D>0` and require `M==mhc_mult`.
- **DESIGN PROPOSAL** — The equality `input.dim(1)==mhc_mult` is mandatory: otherwise the attribute, input layout, loop bound, and GM offsets describe different tensors.
- **DESIGN PROPOSAL** — Set output 0 to `[S,D]` after all checks pass.
- **UNRESOLVED** — The platform does not prescribe an exact error code/message for invalid ranks, attributes, or dimensions; the implementation should use the repository/CANN Host error convention and return non-success consistently.

## 4. InferDataType design

- **PLATFORM FACT** — Legal input and output dtypes are exactly BF16 and FP16, and output dtype equals input dtype in both directions.
- **DESIGN PROPOSAL** — Read input 0 dtype. If it is `DT_BF16` or `DT_FLOAT16`, assign that exact dtype to output 0.
- **DESIGN PROPOSAL** — Return non-success for every other dtype even though the OpDef already restricts registration; explicit validation protects direct or malformed invocations.
- **DESIGN PROPOSAL** — InferDataType does not cast data and does not use `backward`; both modes preserve dtype.

## 5. Baseline task model and multicore split

### 5.1 Independent task

- **DESIGN PROPOSAL** — Define one task as one row-contained hidden-dimension tile `(s,dBegin:dEnd)` of the logical `[S,D]` ownership space.
- **DESIGN PROPOSAL** — Let `tileLength` be the UB tile capacity in elements and `tilesPerRow=ceil_div(D,tileLength)`.
- **DESIGN PROPOSAL** — Let `taskCount=S*tilesPerRow`, computed with checked `uint64_t` arithmetic on Host and recomputed from the validated `s`, `d`, and `tileLength` fields in kernel code. Task `t` maps to `s=t/tilesPerRow`, `tileInRow=t%tilesPerRow`, `dBegin=tileInRow*tileLength`, and `validLength=min(tileLength,D-dBegin)`.
- **DESIGN PROPOSAL** — Keeping every task within one `D` row makes all GM transfers contiguous and makes forward/backward share the same ownership scheme.

### 5.2 Core ownership

- **TEMPLATE FACT** — Multi-tile rows and complete-row shapes with `S>=192` use `usedCoreNum=min(platformAivCoreNum,taskCount)`. The retained rule uses `usedCoreNum=min(platformAivCoreNum,ceil(taskCount/8))` only for complete-row `S<192`, retaining contiguous ownership and 48 cores for wide/medium shapes. It achieved platform-derived `90.9719`; the old unconditional row8 and 32x128 grid were rejected because they reduced wide-shape or platform performance.
- **TEMPLATE FACT** — This rule keeps medium/wide shapes at the full useful AIV count while avoiding excessive scheduling/scalar overhead on small rows. Versus the four-row predecessor, local msprof reduced FP16 forward `[64,256]` from `2.52` to `1.86 us`, forward `[127,513]` from `4.48` to `3.30 us`, and backward `[127,4,513]` from `4.80` to `3.58 us`.
- **DESIGN PROPOSAL** — Divide the contiguous task-ID range by quotient/remainder:

```text
base  = taskCount / usedCoreNum
extra = taskCount % usedCoreNum
coreTaskCount = base + (coreId < extra ? 1 : 0)
coreTaskStart = coreId * base + min(coreId, extra)
coreTaskEnd   = coreTaskStart + coreTaskCount
```

- **DESIGN PROPOSAL** — Each core processes `[coreTaskStart,coreTaskEnd)` sequentially. Core loads differ by at most one task, small shapes naturally use fewer cores, and no core owns an empty range.
- **DESIGN PROPOSAL** — Every logical output element `(s,j)` belongs to exactly one task and one core. Backward therefore needs no atomic operation, cross-core reduction, or synchronization.
- **DESIGN PROPOSAL** — This task split is preferred over a raw flattened `[S*D]` interval because raw intervals can cross row boundaries and complicate the strided `m` offsets.

## 6. Host Tiling design

### 6.1 Proposed minimal TilingData

```cpp
struct MhcExpandTilingData {
    uint64_t s;             // token rows
    uint64_t d;             // hidden dimension
    uint64_t mhcMult;       // replication/reduction extent
    uint32_t tileLength;    // UB tile capacity in elements
    uint32_t usedCoreNum;   // launched AIV cores
    uint32_t mode;          // 0: forward, 1: backward
};
```

| Field | Classification | Why the kernel needs it |
|---|---|---|
| `s` | **DESIGN PROPOSAL** | Computes `taskCount=S*ceil_div(D,tileLength)` and bounds the row domain. |
| `d` | **DESIGN PROPOSAL** | Computes `taskCount`, row/tile mapping, valid tail length, and GM strides. |
| `mhcMult` | **DESIGN PROPOSAL** | Controls forward write count, backward load/add count, and `[S,m,D]` offsets. |
| `tileLength` | **DESIGN PROPOSAL** | Defines UB allocation and normal task capacity; bounded by UB and therefore safely stored as `uint32_t`. |
| `usedCoreNum` | **DESIGN PROPOSAL** | Reconstructs each core's exact task range; bounded by hardware core count. |
| `mode` | **DESIGN PROPOSAL** | Selects the forward or backward process once per core; value is derived from rank on Host. |

- **DESIGN PROPOSAL** — Do not retain the template's `uint32_t length`. All shape products and GM offsets use `uint64_t` to avoid unnecessary narrowing.
- **DESIGN PROPOSAL** — Host computes and validates `taskCount` and input/output total element counts with checked `uint64_t` arithmetic, but does not store them: they are cheaply and unambiguously derived from `s`, `d`, `mhcMult`, and `tileLength` without another potentially inconsistent field.
- **DESIGN PROPOSAL** — Do not store `tilesPerRow`, `taskCount`, per-core starts/counts, total element counts, or `tailLength`; derive each in `uint64_t` from the fields above.
- **DESIGN PROPOSAL** — The final task in every row uses `validLength=min(tileLength,D-dBegin)`, so the tail length is explicit at runtime without a separate field.

### 6.2 Host computation

- **DESIGN PROPOSAL** — Host repeats the same rank, dimension, attribute-consistency, dtype, and overflow checks as inference because Tiling must be safe even if called independently.
- **DESIGN PROPOSAL** — Obtain runtime AIV core count and UB size from `PlatformAscendC`; do not hardcode core count or a nominal UB capacity.
- **DESIGN PROPOSAL** — Set user workspace size to zero. Each task is independent and all temporary data fits in per-core UB.
- **DESIGN PROPOSAL** — Set block dimension to `usedCoreNum`, not all available cores.

## 7. UB planning and tile size

### 7.1 Alignment rule

- **PLATFORM FACT** — On Atlas A2, CANN 8.5 `DataCopyPad` supports GM-to-Local and Local-to-GM non-aligned transfers for FP16/BF16; LocalTensor starts still require 32-byte alignment.
- **DESIGN PROPOSAL** — Allocate every UB buffer in 32-byte multiples and choose normal `tileLength` as a multiple of 16 elements because both public dtypes occupy 2 bytes.
- **DESIGN PROPOSAL** — `tileLength` may exceed `D` for very small shapes; `validLength` controls the logical transfer and vector count.

### 7.2 Forward UB

| Buffer | Classification | Size | Count | Purpose |
|---|---|---:|---:|---|
| `xLocal<T>` | **DESIGN PROPOSAL** | `tileLength*2` bytes | 1 | One contiguous input tile, reused for all `m` output copies. |

- **DESIGN PROPOSAL** — No separate forward output UB is required: the loaded LocalTensor is directly used as the source of each Local-to-GM `DataCopyPad`.
- **DESIGN PROPOSAL** — Forward UB constraint is `Align32(tileLength*2) <= ubSize`.

### 7.3 Backward UB

| Buffer | Classification | Size | Count | Purpose |
|---|---|---:|---:|---|
| `xLocal<T>` | **TEMPLATE FACT** | `tileLength*2` bytes | 2 when `m>1`, otherwise 1 | Current and prefetched-next FP16/BF16 gradient slices. |
| `xFloat<float>` | **TEMPLATE FACT** | `tileLength*4` bytes | 1 except FP16 `m=2` | Current slice converted losslessly to FP32. |
| `accFloat<float>` | **TEMPLATE FACT** | `tileLength*4` bytes | 1 except FP16 `m=2` | FP32 accumulator across `m`. |
| `oLocal<T>` | **DESIGN PROPOSAL** | `tileLength*2` bytes | 1 | One final cast result before GM write. |

- **TEMPLATE FACT** — Retained generic backward UB use is `14*tileLength` bytes for `m>1` because the input queue has two slots; `m=1` uses `12*tileLength` bytes. FP16 `m=2` uses exactly `6*tileLength` bytes: two two-byte input slots and one two-byte output slot, with no unused FP32 buffers.
- **DESIGN PROPOSAL** — Keep `xLocal` and `oLocal` separate in the baseline even though their lifetimes could later be overlapped. Separate storage reduces aliasing and pipeline-order risk.
- **TEMPLATE FACT** — Host uses the matching mode/path coefficient: 12 for `m=1`, 14 for generic `m>1`, and 6 for FP16 `m=2`.
- **DESIGN PROPOSAL** — Host selects the largest positive 16-element multiple satisfying the mode-specific constraint. The backward coefficient is stricter; no arbitrary fixed tile size is hardcoded.
- **DESIGN PROPOSAL** — DataCopy/vector API count limits are also checked before narrowing `tileLength` to `uint32_t`; the UB-derived capacity on the target is expected to be far below those limits.

## 8. Forward kernel dataflow

- **DESIGN PROPOSAL** — Each core iterates only its owned tasks. For task `(s,dBegin,validLength)`, input GM offset is `s*D+dBegin`.
- **DESIGN PROPOSAL** — Copy exactly `validLength*sizeof(T)` bytes from GM to `xLocal` once using `DataCopyPad`.
- **DESIGN PROPOSAL** — For `k=0..mhcMult-1`, write the same local tile to GM offset `(s*mhcMult+k)*D+dBegin` using Local-to-GM `DataCopyPad`.
- **DESIGN PROPOSAL** — This performs one GM read and `m` necessary GM writes per input tile; it avoids rereading the same `x` tile for every replica.
- **DESIGN PROPOSAL** — Forward performs no arithmetic or dtype conversion, so it copies BF16/FP16 bit patterns exactly.

```text
for taskId in owned task range:
    (s, dBegin, validLength) = mapTask(taskId)
    DataCopyPad(xLocal, xGm[s*D + dBegin], validLength * sizeof(T))
    for k in [0, mhcMult):
        DataCopyPad(oGm[(s*mhcMult + k)*D + dBegin],
                    xLocal,
                    validLength * sizeof(T))
```

- **DESIGN PROPOSAL** — A full tile and the last tail tile use the same control flow; only `validLength` differs.
- **DESIGN PROPOSAL** — `S=1,D=1` creates one task, launches one core, transfers one 2-byte logical value through non-aligned `DataCopyPad`, and writes it `m` times without out-of-bounds access.
- **TEMPLATE FACT** — When a full row fits in a tile, the retained forward path batches multiple consecutive owned rows into one GM-to-UB transfer. It writes each replica with one strided UB-to-GM `DataCopyPad`, so the input is still read once and no output element is materialized in an extra UB buffer. Non-fitting rows retain the row-contained tile loop above.

## 9. Backward kernel dataflow

### 9.1 Current correctness-preserving sequence

- **TEMPLATE FACT** — Each task owns one contiguous output tile `o[s,dBegin:dEnd]` and all `m` corresponding input slices.
- **TEMPLATE FACT** — FP16 `m=2` loads both slices into the two-slot input queue, performs one same-dtype vector Add into `oLocal`, and writes it directly. Complete rows use the same strided multi-row batching. This eliminates both input casts, the FP32 Add, and the final cast.
- **TEMPLATE FACT** — For `m=1`, the retained path initializes `accFloat` to positive FP32 zero, casts slice 0 to `xFloat`, and performs `accFloat=0+xFloat`. This preserves the exact signed-zero behavior of the established FP32-sum reference.
- **TEMPLATE FACT** — For `m>=2`, slice 0 is cast directly into `accFloat`; slices `k=1..m-1` are cast to `xFloat` and added serially. This removes one `Duplicate` and one FP32 `Add` per output tile while preserving the remaining accumulation order.
- **TEMPLATE FACT** — The `m>=2` input queue has depth two. Slice 0 is preloaded; after dequeuing slice `k`, the kernel enqueues `k+1` before casting/adding `k`. This overlaps MTE2 with Vector work while preserving the exact serial FP32 reduction order.
- **TEMPLATE FACT** — Complete rows are batched when they fit in UB. Each `k` load uses a strided multi-row `DataCopyPad`, the same rolling prefetch applies, and one core owns all `m` inputs and final outputs for its rows.
- **TEMPLATE FACT** — After all `k`, the kernel casts `accFloat` once to the public dtype with `RoundMode::CAST_RINT`, then writes exactly `validLength` elements to output offset `s*D+dBegin` using `DataCopyPad`.

```text
for taskId in owned task range:
    (s, dBegin, validLength) = mapTask(taskId)
    if mhcMult == 1:
        Duplicate(accFloat, 0.0f, validLength)
        DataCopyPad(xLocal, xGm[s*D + dBegin], bytes)
        Cast(xFloat, xLocal, CAST_NONE, validLength)
        Add(accFloat, accFloat, xFloat, validLength)
    else:
        DataCopyPad(xLocal, xGm[(s*mhcMult)*D + dBegin], bytes)
        Cast(accFloat, xLocal, CAST_NONE, validLength)
        for k in [1, mhcMult):
            DataCopyPad(xLocal,
                        xGm[(s*mhcMult + k)*D + dBegin],
                        bytes)
            Cast(xFloat, xLocal, CAST_NONE, validLength)
            Add(accFloat, accFloat, xFloat, validLength)
    Cast(oLocal, accFloat, CAST_RINT, validLength)
    DataCopyPad(oGm[s*D + dBegin],
                oLocal,
                validLength * sizeof(T))
```

- **TEMPLATE FACT** — The generic loop remains available for every positive `mhc_mult`; BF16 uses it for all multipliers, and FP16 uses it except for the measured `m=2` specialization.
- **DESIGN PROPOSAL** — `DataCopyPad` handles non-32-byte-aligned GM tails. Vector operations use `validLength`, so padded/dummy UB lanes never contribute to an output.
- **DESIGN PROPOSAL** — Each output is finalized by exactly one core, so no atomic add and no inter-core partial buffer are needed.

### 9.2 Same-dtype versus FP32 accumulation

| Strategy | Classification | Advantages | Correctness/performance risks |
|---|---|---|---|
| FP16 direct Add for exactly `m=2` | **TEMPLATE FACT** | Fewer buffers and no casts; lower UB traffic; only one rounding, as in FP32 sum followed by FP16 cast. | A strengthened 262,144-pair finite/extreme test found only `-0+-0` sign-bit divergence (`-0` versus `+0`), which is numerically equal under the platform contract. This reasoning does not extend to `m>2`. |
| BF16 `Add` into BF16 accumulator | **DESIGN PROPOSAL** | Would minimize buffers if available. | CANN 8.5 basic `Add` on Atlas A2 does not support BF16 operands, so this is not a viable common baseline. |
| FP32 accumulator for both public dtypes | **DESIGN PROPOSAL** | BF16/FP16-to-FP32 conversion is lossless; only one final public-dtype rounding; one uniform algorithm; matches the task's referenced TileKernels BF16 backward, which allocates an FP32 fragment. | Uses 12 bytes of UB per logical element and adds Cast/FP32 Add traffic. |

- **DESIGN PROPOSAL** — Keep FP32 accumulation for BF16 and for FP16 `m!=2`; use the measured direct FP16 Add only for exactly two terms. This is based on target API support, rounding equivalence, exhaustive finite-pair evidence, and the platform's numerical correctness contract.
- **PLATFORM FACT** — The target CANN 8.5 Atlas A2 basic `Add` API supports FP16 and FP32 but not BF16; its `Cast` API supports BF16/FP16-to-FP32 and FP32-to-BF16/FP16.
- **TEMPLATE FACT** — The current `m>=2` path initializes from slice 0 and then uses sequential `k=1..m-1` FP32 additions. Do not introduce tree/pairwise reduction before correctness evidence; validation compares the final public-dtype result with `x.float().sum(dim=1).to(dtype)`.
- **DESIGN PROPOSAL** — Use `CAST_NONE` for BF16/FP16 to FP32 and `CAST_RINT` for FP32 to BF16/FP16; CANN 8.5 defines `CAST_RINT` as round-to-nearest, ties-to-even.
- **UNRESOLVED** — The platform does not state an accumulation order/dtype independently of the reference, nor whether hidden cases override default tolerances. The reference-aligned FP32 serial path is the least speculative baseline.

## 10. TilingKey and mode dispatch

| Choice | Classification | Simplicity | Baseline decision |
|---|---|---|---|
| Add forward/backward to TilingKey | **DESIGN PROPOSAL** | Produces more variants and removes the remaining uniform runtime mode branch. | Still deferred pending a focused measurement. |
| Specialize common backward multipliers | **TEMPLATE FACT** | Backward `m=2/4` use compile-time `MHC_MULT_KIND`; forward and every other positive multiplier use key value `0` and the generic runtime fallback. | Retained after build, 44/44 correctness, local A-B-A screening, and platform improvement. |
| Keep runtime `mode` | **TEMPLATE FACT** | The current class still branches once per core for key `0`; specialized backward keys bypass that branch. | Current retained behavior. |

- **TEMPLATE FACT** — Compile-time specialization is now `KernelMhcExpand<DT_X,MHC_MULT_KIND>`, with `DT_X in {FP16,BF16}` and `MHC_MULT_KIND in {0,2,4}`.
- **TEMPLATE FACT** — Host emits a nonzero multiplier kind only for backward `m=2/4`; the public interface and six-field TilingData are unchanged.
- **DESIGN PROPOSAL** — Keep the generic value `0` mandatory so hidden/legal positive multipliers are never narrowed to the published set.

## 11. Correctness risks and required validation matrix

- **PLATFORM FACT** — Required supported coverage includes `S=1`, `D=1`, non-32-byte-aligned `D`, `mhc_mult in {2,4,8}`, BF16, and FP16; the legal multiplier domain is every positive integer.

| Case/risk | Classification | Baseline handling / expected result |
|---|---|---|
| `S=1` | **DESIGN PROPOSAL** | One row; task/core formulas remain valid. |
| `D=1` | **DESIGN PROPOSAL** | One-element tail moved with `DataCopyPad`; vector count is 1. |
| `D` not 32-byte aligned | **DESIGN PROPOSAL** | `validLength` plus non-aligned `DataCopyPad`; no rounded-up GM access. |
| `D<tileLength` | **DESIGN PROPOSAL** | One task per row; `validLength=D`. |
| Last tail tile | **DESIGN PROPOSAL** | `validLength=D-dBegin`; padded lanes are neither reduced nor written. |
| `mhc_mult=2/4/8` | **DESIGN PROPOSAL** | Same positive-integer runtime loop; verify both dtypes and modes. |
| Other positive `mhc_mult`, including `1` | **DESIGN PROPOSAL** | Same loop; `m=1` is identity in both modes and is a useful boundary test. |
| BF16 | **DESIGN PROPOSAL** | Use FP32 accumulation because target basic Add lacks BF16. |
| FP16 | **TEMPLATE FACT** | Use direct same-dtype Add for `m=2`; use FP32 serial accumulation for every other positive multiplier. |
| Missing/null `backward` | **DESIGN PROPOSAL** | Do not dereference; derive mode from rank. |
| `backward` conflicts with rank | **DESIGN PROPOSAL** | Host returns non-success; no kernel launch. |
| Backward `input.dim(1)!=mhc_mult` | **DESIGN PROPOSAL** | Host returns non-success; no ambiguous loop/stride. |
| Invalid dtype | **DESIGN PROPOSAL** | InferDataType/Tiling return non-success. |
| Product/offset overflow | **DESIGN PROPOSAL** | Checked `uint64_t` multiplication rejects before launch. |
| Multi-core race | **DESIGN PROPOSAL** | Unique row-tile output ownership; no atomic/cross-core write. |

- **DESIGN PROPOSAL** — Correctness validation must compare forward materialized output and backward results against the authoritative reference for BF16 and FP16, including small, non-aligned, tail-only, and published `m` cases.
- **TEMPLATE FACT** — Task-local validation covers both dtypes, forward/backward, `m in {1,2,3,4,5,8}`, `S=D=1`, 32-byte boundary neighbors, non-aligned `D`, published medium/large shapes, the declared `mhc_mult=2` default value, rank/mode conflicts, backward `M!=mhc_mult`, invalid rank, and invalid dtype.

## 12. Qualitative performance model

### 12.1 Forward

- **DESIGN PROPOSAL** — Large forward cases are expected to be primarily GM-bandwidth bound: each input element is read once and every required output element must be written `m` times.
- **DESIGN PROPOSAL** — UB read bandwidth also scales with `m` because the same local tile is the source of `m` writes, but no arithmetic is performed.
- **DESIGN PROPOSAL** — For tiny `D`/small task counts, DataCopy setup, address arithmetic, and launch overhead can dominate instead of bandwidth.

### 12.2 Backward

- **DESIGN PROPOSAL** — Large backward cases read `m*S*D` public-dtype elements and write `S*D`, so GM bandwidth remains important.
- **DESIGN PROPOSAL** — Unlike forward, backward also pays `m` input-to-FP32 casts, one final output cast, and `m` FP32 additions/UB accesses per output tile; vector conversion/reduction cost can become material as `m` grows.
- **DESIGN PROPOSAL** — The serial dependency through `accFloat` limits reduction overlap; profiling is required before declaring GM, UB, Cast, or Add the dominant bottleneck.

## 13. Deferred optimization candidates

- **DESIGN PROPOSAL** — After correctness passes, compare aligned `DataCopy` for full tiles with `DataCopyPad` only for tails.
- **DESIGN PROPOSAL** — After correctness passes, evaluate larger/multi-row contiguous transfers and alternative `[S,D]` task shapes to reduce DataCopy setup overhead.
- **DESIGN PROPOSAL** — After correctness passes, consider compile-time `m=2/4/8` unrolling or mode TilingKeys; retain a generic positive-`m` fallback.
- **TEMPLATE FACT** — First-slice accumulator initialization has been measured and retained for `m>=2`; `m=1` deliberately keeps zero initialization for exact signed-zero compatibility.
- **DESIGN PROPOSAL** — After correctness passes, evaluate vector Add scheduling and pairwise/tree reduction while preserving acceptable numerical behavior.
- **DESIGN PROPOSAL** — After correctness passes, evaluate reuse/aliasing of `xLocal` and `oLocal` to increase tile capacity.
- **DESIGN PROPOSAL** — After correctness passes, evaluate single/double buffering and copy/cast/add overlap.
- **DESIGN PROPOSAL** — After correctness passes, tune `tileLength`, core/task partitioning, and small-shape core counts with measured evaluator/profile evidence.
- **DESIGN PROPOSAL** — Each candidate must change one major dimension and pass guard/build/validate before benchmark; no optimization may narrow dtype, shape, mode, or positive-`m` coverage.

## 14. Baseline implementation and evidence

- **TEMPLATE FACT** — InferShape, InferDataType, Host Tiling, the six-field TilingData, forward copy path, generic backward FP32 accumulation path, and FP16 `m=2` direct-Add specialization are implemented in `workspace/code/` according to Sections 3–10. The public OpDef, filenames, directory layout, and dtype-only TilingKey remain unchanged.
- **TEMPLATE FACT** — The forward kernel uses one `TQueBind<VECIN,VECOUT,1>` tile, batches complete rows when possible, and reuses the same LocalTensor for all `m` GM writes. Generic backward uses a two-slot input queue for `m>1`, a one-slot output queue, FP32 converted/accumulator buffers, complete-row batching, and rolling next-slice prefetch. FP16 `m=2` retains the two input slots and output slot but omits the unused FP32 buffers. There is no mode TilingKey and no user workspace.
- **TEMPLATE FACT** — Task-local `scripts/build.sh` performs configure, kernel/Host/ACLNN binary generation, and package generation. It also creates a temporary read-only OPP view when the installed vendor `config.ini` is not readable; it does not alter the system installation.
- **TEMPLATE FACT** — A real local build completed successfully with CANN `9.0.0-beta.1`: both FP16/BF16 kernel variants, Host tiling library, generated ACLNN library, and `.run` package were produced. This is local compiler evidence only; the platform target remains CANN `8.5.0` on `ascend910b`.
- **TEMPLATE FACT** — A physically separate user-level CANN `8.5.0` Toolkit plus `Ascend-cann-910b-ops` environment is now installed. A clean build in that isolated environment also produced both kernel variants, Host tiling library, ACLNN library, and `.run` package; no CANN 9-only source compatibility error was found.
- **TEMPLATE FACT** — The generated ACLNN signature exposes `x`, `mhcMult`, `backward`, and `out`, so the task-local test can exercise explicit mode values and `mhc_mult=2`, but cannot truly omit the `backward` argument through that generated entrypoint.
- **TEMPLATE FACT** — Task-local correctness validation now calls the generated ACLNN API directly through ACL device memory and uses NumPy references; it does not require torch or torch-npu.
- **TEMPLATE FACT** — Independent-process lifecycle probes established that devices 2 and 4 are runtime-stable under both CANN 8.5 and CANN 9.0. All 16 requested probes completed within 30 seconds. A first `aclrtSetDevice` can take about 9–11 seconds, while warm independent probes typically take about 0.4–0.6 seconds; this is environment initialization and not operator latency.
- **TEMPLATE FACT** — NNOP diagnostics identified the earlier built-in ACLNN failure as a product-OPS mismatch: the physical `Ascend910_9382` device requests `ascend910_93`, while the competition build root intentionally provides `Ascend-cann-910b-ops` and `ascend910b`. A separate CANN 8.5 + `Ascend-cann-A3-ops` root makes the built-in `aclnnAbs` full lifecycle pass; no MhcExpand source change was required for that environment fix.
- **DESIGN PROPOSAL** — Maintain one persistent algorithm source in `workspace/code/`, always targeting competition CANN 8.5 + 910B OPS + `ascend910b`. Use `scripts/build-platform-910b.sh` as the mandatory first gate after every algorithm change.
- **DESIGN PROPOSAL** — Use `scripts/validate-local-a3.sh` as the second gate. It creates a clean temporary mirror and adapts only `ASCEND_COMPUTE_UNIT` and OpDef `AddConfig` from `ascend910b` to the runtime-confirmed `ascend910_93`. Its recursive scope guard rejects every other source difference, so no second long-lived algorithm implementation exists.
- **TEMPLATE FACT** — On 2026-08-21, the final formal CANN 8.5 + 910B OPS clean build generated Host tiling, ACLNN library, both dtype kernels, and an `ascend910b` package: **PLATFORM 910B BUILD PASS**.
- **TEMPLATE FACT** — The final clean A3 mirror build generated an `ascend910_93` package, installed it into a temporary root, loaded `libcust_opapi.so`, and obtained non-null executors. The 44-case suite passed exactly, covering both directions/dtypes, `m in {1,2,3,4,5,8}`, aligned and non-aligned dimensions, published medium/large shapes, default-attribute behavior, and invalid-contract rejection: **LOCAL A3 CORRECTNESS PASS**.
- **PLATFORM FACT** — Seven CANNJudge submissions reached `Pass`; all five entries in every submission passed with `precision_ratio=1`. The retained row8-below-`S=192` version has public platform times `[2.26,22.54,1379.98,47.18,2.38]` and derived mean score `90.9719`.
- **TEMPLATE FACT** — The task-local JSONL contains 42 cases: 38 default correctness cases, 22 performance-tagged cases, and four opt-in published stress cases excluded from routine multi-gigabyte allocation.
- **TEMPLATE FACT** — The retained `m>=2` first-slice initialization reduced the four large backward local-proxy medians by 2.90%-7.52% versus the mean of two baseline runs (mean 4.93%). Candidate and final evidence is recorded in `optimization-log.md`.
- **TEMPLATE FACT** — Moving `TPipe` outside the class was rejected as noise-level; the backward `m=1` direct-copy candidate was rejected because it changed BF16 negative-zero bits relative to the exact FP32-sum reference.
- **TEMPLATE FACT** — FP16 `m=2` same-dtype Add was initially rejected under an unnecessarily bit-exact policy, then retained after reconciling the sole `-0 + -0` sign-bit difference with the platform's numerical contract. The strengthened 262,144-pair finite/extreme test found no other mismatch. Device profiling measured `[64,2,256]` at `1.76 us` versus `1.86 us`; matching the actual six-byte UB footprint further improved `[256,2,7168]` from `7.06` to `6.32 us` and `[1024,2,4096]` from `9.24` to `8.60 us`.
- **PLATFORM FACT** — CANNJudge returned per-case times and `best_time` values but no aggregate score field. The current documented-formula derived mean is `90.9719`; it is platform-result-derived evidence, not a directly returned score field.
- **UNRESOLVED** — The platform result does not map its five opaque entries to public shapes, so optimization decisions must not infer hidden testcase contents from their timing order.

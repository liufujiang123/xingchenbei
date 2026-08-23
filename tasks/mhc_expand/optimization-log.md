# MhcExpand optimization log

Only executed experiments are recorded here. All latency values are CANN 8.5 ACL-runtime-event measurements on the local `Ascend910_9382` A3 device. They are a same-device optimization proxy, not CANNJudge 910B scores. The event interval can include stream idle time caused by host enqueue pacing; absolute effective-bandwidth values are therefore descriptive only.

## Evaluation matrix

- Correctness matrix: 38 valid tensor cases plus 6 API/invalid-contract cases, for 44 pytest cases total.
- Directions/dtypes: forward and backward, FP16 and BF16.
- Multipliers: `m in {1,2,3,4,5,8}`; published performance sizes use `m in {2,4,8}`.
- Hidden dimension boundaries: `D in {1,16,17,31,32,33,129,256,513,4096,7168}`.
- Shape classes: single element, 32-byte boundary neighbors, non-aligned rows, small/medium/large published shapes, and separately tagged 8192x7168 stress cases.
- Performance subset: 22 cases, warmup 5, 7 active samples, case-dependent repeated launches, median ACL event time.
- Source of truth: `tests/mhc_expand_perf_cases.jsonl`; the four stress cases are opt-in and are not allocated by the default correctness/performance run.

## Measured candidates

| Candidate | Focused hypothesis | Gates | Local A3 result | Decision |
|---|---|---|---|---|
| Baseline | Establish an executable reference and representative case matrix. | CANN 8.5 910B build PASS; A3 42/42 PASS; 20 perf cases measured. | Large backward medians: 25.446-25.546 us (`256x8x7168`) and 29.466-29.522 us (`1024x4x4096`). | Reference |
| C1: external `TPipe` | Moving `TPipe` from the kernel object to the entry function may reduce Scalar constant-propagation overhead. | 910B build PASS; A3 42/42 PASS. | Forward mean +0.66%, backward mean +0.52%; only 6/10 cases won in each direction and large cases regressed about 2% in places. The effect was inside run-to-run noise. | REJECT; reverted |
| C2: initialize accumulator from first slice | For backward `m>=2`, casting slice 0 directly into FP32 accumulator removes one `Duplicate` and one FP32 `Add` per output tile without changing the remaining serial accumulation order. | 910B build PASS; A3 42/42 PASS; A-B-A proxy comparison completed. | Four large backward cases all improved; A-B-A candidate run improvement was 1.37%-8.22%, mean 5.24%. Final retained run versus the mean of two baseline runs improved 2.90%-7.52%, mean 4.93%. | KEEP for `m>=2` |
| C3: backward `m=1` copy path | A one-term reduction could bypass all casts/adds and use the forward copy path. | 910B build PASS; A3 43/44, BF16 m=1 failed exact-bit regression. | 1.51% FP16 and 3.91% BF16 proxy gains, but one BF16 `-0` (`0x8000`) remained negative while the FP32-sum reference produced `+0` (`0x0000`). | REJECT; reverted |
| Final retained | C2 for `m>=2`; preserve baseline zero-initialized reduction for `m=1`. | CANN 8.5 910B Host/FP16/BF16/ACLNN/package PASS; A3 44/44 PASS; 22 perf cases measured. | Large backward mean improvement 4.93% on the local proxy; m=1 changed by -0.52% FP16/+0.71% BF16, i.e. noise-level and bit-exact correctness preserved. | KEEP |

## Commands and evidence

```text
tasks/mhc_expand/scripts/build-platform-910b.sh
MHC_EXPAND_DEVICE_ID=4 tasks/mhc_expand/scripts/validate-local-a3.sh
MHC_EXPAND_DEVICE_ID=4 tasks/mhc_expand/scripts/bench-local-a3.sh <label>
```

- Baseline: `runs/performance/mhc_expand_baseline.md`, `mhc_expand_baseline2.md`, and `mhc_expand_baseline_m1.md`.
- Candidate reports: `runs/performance/mhc_expand_candidate1_tpipe.md`, `mhc_expand_candidate2_first_acc.md`, and `mhc_expand_candidate3_m1.md`.
- Final report: `runs/performance/mhc_expand_final.md`.
- Final build/correctness logs: `runs/performance/final-platform-build.txt` and `final-correctness.txt`.

## CANNJudge platform result

- Exactly one submission was issued: `6a88808d82cffa8f167f1b9d` for verified MhcExpand problem `6a7c1a74a52e0f540a89d39b`.
- Platform terminal status: `Pass`; all five result entries passed with `precision_ratio=1`.
- Reported per-case times: `4.84`, `21.78`, `1381.96`, `58.28`, and `4.88`.
- Reported `best_time` values: `2.02`, `15.22`, `1370.10`, `46.66`, and `2.58`.
- The platform response had no aggregate score field. The documented formula gives per-case contributions `41.7355`, `69.8806`, `99.1418`, `80.0618`, and `52.8689`, with derived mean `68.7377`.
- Sanitized evidence: `runs/cannjudge/20260821T164852Z-6a88808d82cffa8f167f1b9d.json`. It stores no credentials, testcase IDs, or testcase contents.
- Harness issue found: `tools/cannjudge_eval.py` treats `Accepted` as terminal but not the actual returned status `Pass`; this does not affect the submission result but prevents automatic prompt evidence finalization.

## Remaining hypotheses

- Use the new CANNJudge per-case timing gaps, plus profiler evidence if a physical compatible target becomes available, before changing the forward path; A3 repeated-launch data alone cannot identify platform HBM limits.
- Evaluate aligned `DataCopy` for full tiles and `DataCopyPad` only for tails.
- Evaluate compile-time `m=2/4/8` unrolling with a generic positive-`m` fallback.
- Evaluate double buffering only after profiling proves copy/vector overlap can hide material latency.
- Revisit core/task split and tile size using actual 910B evidence, especially small `S` and very wide `D`.

## Subsequent optimization campaign

The following work continued from the retained first-slice baseline. Device-side
numbers are medians from `msprof` `Task Duration(us)` on the same local A3; they
are more reliable for kernel A/B decisions than the ACL event aggregate, whose
host-side executor pacing was observed to drift by several microseconds.

| Candidate | Focused hypothesis | Gates/evidence | Decision |
|---|---|---|---|
| Forward multi-row batching | One GM read for several complete rows plus strided `m` writes reduces per-row DMA setup. | 910B build PASS; A3 44/44 PASS; two local repeats; CANNJudge submission `6a8886ec82cffa8f167f907a` Pass 5/5. | KEEP |
| Backward multi-row batching + rolling MTE2 prefetch | Batch complete rows and enqueue slice `k+1` before cast/add of slice `k` to overlap GM input with Vector work without changing FP32 serial accumulation order. | 910B build PASS; A3 44/44 PASS; CANNJudge submission `6a888abc82cffa8f167fe54c` Pass 5/5. | KEEP |
| Forward two-slot queue | Two input slots might overlap consecutive row batches. | Build/correctness PASS; local proxy `312.383 us`, slower than retained. | REJECT; reverted |
| Backward output/converted-buffer alias | Reusing UB could enlarge tiles and reduce batches. | Build/correctness PASS; local proxy `310.096 us`, slower than retained. | REJECT; reverted |
| Generic 32 KiB traffic-aware core count | Launch fewer cores for small transfers. | Build/correctness PASS; local proxy `305.015 us`, slower than retained. | REJECT; reverted |
| Aligned `DataCopy` fast path | Avoid `DataCopyPad` on aligned full transfers. | 910B build PASS; A3 44/44 PASS; two proxy runs `303.650` and `303.291 us` versus retained `302.678 us`. | REJECT; reverted |
| Forward `TBuf` with explicit MTE2/MTE3 events | Hand-managed copy events might reduce queue overhead. | 910B build PASS; A3 44/44 PASS; proxy `313.740 us`. | REJECT; reverted |
| Four complete row tasks per core | Fewer AIV cores should reduce small-shape scheduling/scalar overhead without reducing wide-shape parallelism. | 910B build PASS; A3 44/44 PASS; proxy repeats `296.803` and `296.166 us`; msprof confirmed device-side gains. | Superseded by 32x128 grid |
| Eight complete row tasks per core | Further reduce small-shape core count. | 910B build PASS; A3 44/44 PASS; proxy `308.278 us`; msprof showed small gains but wide backward regression after blockDim fell to 32. | REJECT; reverted |
| Public 32x128 reference grid for blockDim | Use `min(ceil(S/32)*ceil(D/128), S, AIV cores)` as the row-path parallelism while retaining contiguous row ownership. | 910B build PASS; A3 44/44 PASS. Local msprof improved FP16 forward `[64,256]` `5.04 -> 1.62 us` and `[127,513]` `5.62 -> 3.54 us`. The first platform submission containing the grid passed 5/5 but returned `[4.78,22.38,1380.58,47.44,4.98]`, derived `71.6476`, below the prior no-grid `74.0741`. | REJECT after platform A/B; reverted |
| Grid D tile 512 | Fewer cores might further help small row-path work. | 910B build PASS; A3 44/44 PASS; msprof regressed `[64,256]` to `1.74 us` and `[127,513]` to `4.02 us`. | REJECT; reverted |
| Grid D tile 160 | Preserve 4 cores for `[64,256]` and use 16 for `[127,513]`. | 910B build PASS; A3 44/44 PASS; no consistent gain over D tile 128, including backward `3.74` versus `3.68 us`. | REJECT; reverted |
| FP16 backward `m=2` direct Add | A same-dtype Add could remove two input casts and one output cast. | 910B build PASS; ordinary A3 44/44 passed; msprof `[64,2,256]` `1.86 -> 1.74 us`. A strengthened 262,144-pair finite/extreme bit test then found `-0 + -0` produced FP16 `0x8000`, while the established FP32-sum reference produced `0x0000`. | REJECT; reverted |
| Grid S tile 64 | Fewer cores might further reduce launch/scalar overhead while the D grid preserves wide-shape parallelism. | 910B build PASS; A3 44/44 PASS; msprof small forward `1.62 -> 1.64 us`, non-aligned forward `3.54 -> 3.50 us`, and non-aligned backward `3.68 -> 3.76 us`. | REJECT; reverted |
| FP16 backward `m=2` numerical-contract reconsideration | For exactly two terms, same-dtype Add and FP32-sum-then-cast have the same finite numerical result; signed-zero polarity is not a platform numerical error. | 910B build PASS; A3 44/44 PASS; the strengthened 262,144-pair finite/extreme test again found exactly one bit mismatch, `-0+-0`, with zero absolute error. Device msprof measured `[64,2,256]` `1.86 -> 1.76 us`. | KEEP |
| FP16 `m=2` two-core small split | Two instead of four cores might reduce scheduling overhead after direct Add shortened the vector path. | 910B build PASS; A3 44/44 PASS; msprof `[64,2,256]` regressed `1.76 -> 1.84 us`. | REJECT; reverted |
| FP16 `m=2` actual UB footprint | Removing unused FP32 buffers and using the true 6-byte coefficient should permit multi-row batches for large D. | 910B build PASS; A3 44/44 PASS; msprof `[64,2,256]` `1.76 -> 1.72 us`, `[256,2,7168]` `7.06 -> 6.32 us`, and `[1024,2,4096]` `9.24 -> 8.60 us`. | KEEP |
| Remove 32x128 grid, retain FP16 `m=2` path | Isolate the locally promising FP16 specialization from the platform-regressing core grid. | 910B build PASS; A3 44/44 PASS; platform Pass 5/5 with `[4.12,22.36,1385.28,47.76,4.34]`, derived `74.3072`, above the prior `74.0741`. | KEEP as platform baseline |
| Four complete-row tasks per core, revisited | A moderate 16/32-core split may retain the grid's local scheduling benefit without its aggressive 4/20-core platform regression. | 910B build PASS; A3 44/44 PASS; platform Pass 5/5 with `[2.78,22.42,1375.78,47.34,2.80]`, derived `85.6940`. | KEEP as new platform baseline |
| Eight complete-row tasks per core below S=192 | The prior row8 local small-shape gains can be retained while an explicit threshold prevents its known 256-row wide-shape 32-core regression. | 910B build PASS; A3 44/44 PASS. Versus row4, msprof forward `[64,256]` `2.52 -> 1.86 us`, forward `[127,513]` `4.48 -> 3.30 us`, and backward `[127,4,513]` `4.80 -> 3.58 us`; `S>=192` remains 48-core. Platform Pass 5/5 with `[2.26,22.54,1379.98,47.18,2.38]`, derived `90.9719`. | KEEP as new platform baseline |
| Single-slot two-row generic reduction at the UB capacity boundary | If the two-slot rolling-prefetch tile holds only one row but a single input slot holds two, doubling row batch size might outweigh lost MTE2/Vector overlap. | 910B build PASS; A3 44/44 PASS. The local event proxy regressed wide FP16/BF16 backward from the retained `13.87/14.26 us` range to `20.98/21.05 us`. Two `msprof` attempts blocked before custom-op loading in `hdcdrv_recv_peek_wait`, so they supplied no kernel evidence. | REJECT; reverted without platform submission |
| Two FP32 accumulators for wide `m>=4` backward | Splitting even/odd `k` slices between two accumulators could shorten the serial Vector Add dependency chain while retaining rolling MTE2 prefetch. | 910B build PASS; A3 44/44 PASS. The local event proxy measured wide FP16/BF16 backward at `16.17/16.54 us`, slower than the retained `13.87/14.26 us` range; the extra final Add and UB footprint outweighed the shorter chain. | REJECT; reverted without platform submission |
| BF16 backward `m=2` direct Add | As with FP16, a two-term BF16 sum is exactly representable in FP32 before final BF16 rounding, so direct Add could remove conversion traffic without changing finite numerical values. | CANN 8.5 910B build failed in `kernel_operator_vec_binary_impl.h`: the instantiated `vadd` requires `__ubuf__ half *`; the public BF16 kernel cannot use this API path. | REJECT at build gate; reverted |
| Reuse the final generic input queue slot as output | Binding the two-slot VECIN queue to VECOUT could remove the separate output slot, enlarge the generic tile, and preserve rolling prefetch. | The corrected 910B build passed, but A3 validation failed first on BF16 `m=2`: the result equaled the final input slice instead of the FP32 reduction. An earlier shared-queue version also corrupted FP16 `m=2`; isolating that fast path did not make generic slot reuse valid. | REJECT at correctness gate; reverted without benchmarking or platform submission |
| Compile-time backward `m=2/4/8` | Removing the runtime reduction bound and offset branches for published multipliers might reduce Scalar overhead. | 910B build and A3 44/44 passed. Two candidate event runs totaled `309.26/307.21 us` versus a same-window baseline `316.70 us`, but FP16 wide `m=8` regressed from `14.24 us` to `14.96/16.38 us`; `msprof` again blocked before operator loading in `hdcdrv_recv_peek_wait`. | REJECT the `m=8` specialization; no platform submission |
| Compile-time backward `m=2/4`, generic `m=8` fallback | Retain the repeatable small/non-aligned `m=4` code-generation benefit without exposing wide `m=8` to its observed regression. | 910B Host/FP16/BF16/ACLNN/package PASS; A3 44/44 PASS; local proxy `300.35 us`. Platform Pass 5/5 with `[2.06,22.68,1376.62,47.60,2.32]`, best times `[2.02,15.22,1366.96,46.66,2.32]`, derived `92.4979`. | KEEP as new platform baseline |
| Compile-time forward/backward mode | Removing the remaining per-core mode branch and unrelated mode body might help forward and generic backward keys. | 910B build and A3 44/44 passed. Local proxy was `301.89 us` versus retained `300.35 us`; FP16/BF16 directions conflicted on both medium `m=4` and wide `m=8`. | REJECT; reverted without platform submission |
| Forward `m=4` whole-row UB materialization | Loading several complete rows once and materializing four contiguous copies in UB might replace strided GM stores with longer output transfers. | The candidate was reverted after the A3 correctness gate first failed the visible non-aligned BF16 forward case with `D=33`; it was not benchmarked or submitted. | REJECT; reverted |
| Light complete-row scheduling (`C1`) | Forward and FP16 backward `m=2` have much less Vector/UB work than generic backward. For complete rows with `(m+1)*D*2 <= 2 KiB`, assigning about 16 rather than eight rows per core may reduce small-transfer scheduling and Scalar overhead without changing the heavier paths. | Fresh CANN 8.5 + 910B Host/FP16/BF16/ACLNN/package build PASS; local A3 44/44 PASS. Targeted same-window A3 `msprof` improved three selected paths by 12.7%-15.1%. The single authorized CANNJudge submission passed 5/5 precision but returned `[4.48,22.48,1384.70,47.26,4.94]`, versus the retained row8 baseline `[2.06,22.68,1376.62,47.60,2.32]`; the first and fifth opaque entries regressed by about 2x. | REJECT after platform A/B; reverted |

### Light complete-row scheduling evidence

- The generic diagnosis had no target profile and therefore supplied only static
  memory/bandwidth/tiling/pipeline risk tags. The focused hypothesis instead came
  from the retained row4/row8 platform history and the current per-path traffic
  model; the static tags are not reported as measured bottlenecks.
- The retained default remains eight complete rows per core below `S=192`.
  The rejected candidate switched only forward or the FP16 backward `m=2`
  direct-Add path to 16 when a row's mandatory input/output GM traffic was at
  most `2 KiB`; no part of that policy remains in the source.
- Two candidate ACL-event reports totaled `292.364397` and `302.813802 us` for
  the 22-case matrix. Retained binaries measured `308.704934 us` before and
  `314.710532 us` after. Because unchanged cases moved in the same direction,
  these aggregate values are descriptive only; the path-specific device-task
  medians above are the promotion evidence.
- The task-local ACL benchmark accepts repeatable `--case-id` filters so a
  profiler run can isolate selected JSONL cases. Omitting the option preserves
  the existing complete-matrix behavior.
- The single authorized platform evaluation was submission
  `6a89b26d82cffa8f16910e53`. It reached `Pass` with `precision_ratio=1` on all
  five entries and times `[4.48,22.48,1384.70,47.26,4.94]`; current
  `best_time` was `[2.02,15.22,1366.96,46.66,2.10]`, giving derived mean
  `70.5507`. The row16 policy was therefore reverted. No second submission was
  issued.

### Post-row16 recovery candidates

| Candidate | Hypothesis | Evidence | Decision |
|---|---|---|---|
| Compile-time Init specialization for existing backward `m=2/4` keys | Constant-folding queue/buffer initialization may remove Scalar setup. | 910B build PASS; A3 44/44 PASS. Targeted A-B-A `msprof` changed FP16/BF16 `m=2` by +1.18%/+1.66% and `m=4` by -0.30%/-0.90%, all noise-level; unchanged control moved +3.11%. | REJECT; reverted |
| Forward aligned `m=2` UB materialization | Two UB copies followed by one contiguous MTE3 write may beat two strided MTE3 descriptors. | 910B build PASS; A3 44/44 PASS. FP16 and BF16 forward `m=2` both changed `1.58 -> 1.60 us` while an unchanged control moved `1.78 -> 1.76 us`. | REJECT; reverted |
| Backward `m=1` Cast + scalar zero Add | Cast directly into the FP32 accumulator and use in-place `Adds(+0)` to preserve negative-zero normalization while removing `Duplicate` and a binary-Add UB source. | 910B Host/FP16/BF16/ACLNN/package PASS; A3 44/44 PASS. A-B-A device-task medians were FP16 `5.84/5.14/5.56 us` and BF16 `5.90/5.16/5.62 us`; the unchanged control was `1.64/1.56/1.58 us`. Platform submission `6a89c15882cffa8f1692f149` passed 5/5 but measured `[2.12,22.62,1373.38,47.68,2.34]`, derived `89.9411` versus retained-baseline `90.6013` under the same current best vector. | REJECT after platform isolation; reverted |
| Backward `m=1` dedicated key and true 8-byte UB footprint | Removing the now-unused converted buffer raises complete-row batch capacity from four to six for `D=4096`, potentially making each core finish in one batch. | 910B build PASS; A3 44/44 PASS. Against the retained `5.14/5.16 us`, two repeats measured FP16/BF16 at `5.90/5.98` and `5.40/5.52 us`; both remained slower after control normalization. Larger batches and the dedicated code shape outweighed the saved batch boundary. | REJECT; fully reverted |
| Backward cross-batch first-slice prefetch | Enqueue the next complete-row batch's `k=0` input before the current batch's final output Cast to overlap one additional MTE2 transfer with Vector work. | 910B build PASS; A3 44/44 PASS. A-B-A `msprof` measured FP16 `12.48/12.42/12.28 us` and BF16 `12.44/12.38/12.26 us`; controls stayed at `1.64 us` and `3.28-3.30 us`. The candidate was about 0.3% slower than the two-side baseline mean. | REJECT; fully reverted |
| Omit GlobalTensor binding lengths | Removing four Init-time size calculations and two length arguments may reduce Scalar setup on tiny paths; actual accesses remain bounded by Host Tiling. | 910B build PASS; A3 44/44 PASS. A-B-A small-path timings were FP16 forward `1.64/1.58/1.58 us`, BF16 forward `1.64/1.56/1.56 us`, FP16 backward `1.74/1.66/1.66 us`, and BF16 backward `1.86/1.76/1.76 us`. Candidate and after-baseline were identical, so the apparent movement was temporal drift rather than attributable gain. | REJECT; explicit binding bounds restored |
| Compile-time forward `m=2/4/8` | Removing the runtime forward replication bound for the published multipliers might reduce Scalar issue overhead. | Harness baseline `299.8188 us`; candidate `300.853668 us` after guard, 910B build, A3 44/44 validation, and the full local performance matrix. | REJECT by Harness; fully reverted |
| Large expanded-row sequential writes | Profiling showed forward dominated by MTE3, so disabling multi-row strided batching for expanded rows at least 32 KiB might improve sequential GM stores. | Harness guard/build/A3 44/44 PASS, but the full local matrix regressed to `305.1254 us` from the retained Harness baseline `299.8188 us`. | REJECT by Harness; fully reverted |
| Explicitly unroll compile-time backward `m=4` | The measured Scalar ratio near `0.44` might come from the reduction loop and address generation even though MTE2/Vector overlap is already effective. | Harness guard/910B build/A3 44/44 PASS. The aggregate proxy was `301.4438 us`; same-device profile changed FP16/BF16 `[1024,4,4096]` from `12.16/12.30 us` to `12.94/12.92 us`, while Scalar remained `0.443/0.440`. | REJECT by Harness/profile; the compiler-generated loop was not the bottleneck and the explicit code was fully reverted |
| Preload both backward `m=4` input slots at startup | Harness classified the path as MTE2/Vector-heavy, so filling both queue slots before the first Cast might reduce the startup bubble. | 910B build PASS. A3 validation stalled in `backward_boundary` before completion: after dequeuing `k0`, the code attempted to allocate `k2` before freeing the physical `k0` slot, so the two-slot queue had no allocatable buffer. The agent-owned validation process was stopped rather than waiting for its 300-second timeout. | REJECT at correctness/synchronization gate; fully reverted and not benchmarked or submitted |
| Corrected two-slice startup preload | Compute and free the current slot before enqueuing `k+2`, preserving one-slice lookahead without over-allocating the two-slot queue. | Harness guard/910B build/A3 44/44 PASS, but the aggregate proxy was `311.473464 us`. Same-device profile changed FP16/BF16 `[1024,4,4096]` from `12.16/12.30 us` to `12.38/12.44 us`, and Scalar rose to `0.454`. | REJECT by Harness/profile; fully reverted and not submitted |
| Forward `m=4` per-core output phase shift | Desynchronizing the four independent output-copy phases across cores might reduce synchronized GM write contention in the MTE3-bound medium path. | Harness guard/910B build/A3 44/44 PASS. The full proxy was `294.332667 us` versus the same-device `289.790065 us` baseline. Target FP16/BF16 changed `13.390/13.626 -> 13.296/13.748 us`; a later targeted repeat changed `15.000/15.252 -> 14.896/15.718 us`, again showing a dtype conflict rather than a robust gain. | REJECT; fully reverted and not submitted |
| Forward aligned `m=4` contiguous UB materialization | Trading idle Vector/UB bandwidth for one contiguous output transfer might remove the strided MTE3 pattern. | `Adds` was rejected at 910B build because CANN 8.5 does not support BF16 `Adds`. The supported UB `DataCopy` version exposed a missing MTE2-to-Vector queue event at 42/44; a dedicated input queue fixed correctness to 44/44. The correct version still regressed the full proxy to `307.362997 us` and target FP16/BF16 to `15.038/15.322 us` from `13.390/13.626 us`. | REJECT; fully reverted and not submitted |
| Forward `m=4` contiguous input-row descriptor | Aligned input rows are contiguous in both GM and UB, so one long MTE2 block might be cheaper than a zero-stride multi-block descriptor. | Harness guard/910B build/A3 44/44 PASS. The full proxy was `305.599665 us`; target FP16/BF16 measured `14.656/14.330 us`, with no evidence of improvement after accounting for run-to-run device drift. | REJECT; fully reverted and not submitted |
| Forward `m=4` pipelined contiguous-output materialization | Two output slots might overlap UB replication with the previous contiguous MTE3 write and recover the cost seen in the single-slot materialization. | Harness guard/910B build/A3 44/44 PASS, but the full proxy was `308.442870 us`; target FP16/BF16 were `14.512/14.904 us`. The extra UB copy work and smaller batches remained more expensive than direct strided output. | REJECT; fully reverted and not submitted |
| Forward `m=4` two-slot bound copy queue | Halving the row batch and using two original GM-to-UB-to-GM bound slots might overlap next-batch MTE2 with prior-batch MTE3 without extra traffic. | Harness guard/910B build/A3 44/44 PASS. The full proxy was `292.281400 us` versus the adjacent retained `284.769067 us`; target FP16/BF16 were `14.240/14.590 us` versus `13.920/14.314 us`. The second descriptor batch outweighed the achieved overlap. | REJECT; fully reverted and not submitted |
| Forward `m=4` interleaved row ownership | Cyclic rows could make same-phase writes from adjacent cores target adjacent logical output rows and improve GM channel utilization. | Harness guard/910B build/A3 44/44 PASS. The full proxy was `299.354203 us`; target FP16/BF16 were `14.138/14.270 us` versus the adjacent retained `13.920/14.314 us`, a FP16 regression and noise-level BF16 change. | REJECT; fully reverted and not submitted |

- No CANNJudge submission was made for the rejected recovery candidates. The
  retained `m=1` candidate passed every local gate, but its first Harness
  platform attempt stopped before login with `missing CANNJUDGE_EMAIL`; no
  submission ID was created and no platform quota was consumed. The user has
  subsequently authorized ten new CANNJudge submissions. The recovered direct
  workflow created `6a89c15882cffa8f1692f149`, so **1/10** has been consumed
  under that authorization and nine remain.

### Measured Harness bottleneck diagnosis

- The first profiled `diagnose` attempt on device 4 failed in
  `aclrtSetDevice` before loading the custom operator. Harness recorded
  `reject:profile_failed`; it is environment evidence, not a kernel
  bottleneck result.
- The identical profile command then completed on device 2. Harness record
  `runs/harness/20260822T155157Z-measured-bottleneck-device2.json` reports
  `passed:diagnosis`, confidence `profile_observed`.
- Forward `[1024,4096],m=4` measured `12.72-12.74 us`, with MTE3 `0.643`,
  MTE2 `0.247`, Scalar `0.125`, and Vector `0.002`. Forward
  `[256,7168],m=8` measured `11.40 us`, with MTE3 `0.648`, MTE2 `0.227`,
  Scalar `0.146`, and Vector `0.002`. The retained forward path is therefore
  primarily an output-copy/GM-bandwidth path; a Cube reformulation has no
  measured justification.
- Backward `[1024,4,4096]` measured `12.16-12.30 us`, with MTE2
  `0.767-0.773`, Vector `0.616-0.648`, Scalar `0.436-0.440`, and MTE3
  `0.176-0.179`. Backward `[256,8,7168]` FP16 measured `13.08 us`, with
  MTE2 `0.814`, Vector `0.566`, Scalar `0.608`, and MTE3 `0.100`.
  MTE2/Vector overlap is already material; the next focused target is reducing
  Scalar loop/address issue without removing the rolling input pipeline.
- The diagnosis path emitted the candidate families
  `vector.mte_v_overlap`, `vector.reduce_pass_fusion`,
  `vector.scan_blocking`, `vector.instruction_fusion`, and
  `common.working_set_liveness`. These are planning inputs, not proof that any
  particular code rewrite will improve performance.

### Reproducible CANNJudge submission path

- The two previously successful direct submissions prove the supported path:
  `python3 tools/agent_loop.py platform --task mhc_expand --name <candidate> --submit`.
  The task config supplies the verified problem ID and submission source, and
  the Harness invokes the official repository-local CANNJudge adapter. A
  ChatGPT connector, Work Agent, or browser workflow is not part of this path.
- The invoking process must inherit `CANNJUDGE_EMAIL` and either
  `CANNJUDGE_CIPHERTEXT_FILE` or `CANNJUDGE_CIPHERTEXT`. The private key is
  resolved by the official skill unless `CANNJUDGE_PRIVATE_KEY` overrides it.
  Plaintext passwords remain prohibited, and no credential value is stored in
  the repository or Harness record.
- A platform run is counted against the authorization only after the adapter
  prints `CANNJUDGE_SUBMISSION_ID=...`. A credential preflight failure before
  that line is not a submission.

### Pipeline and matrix-path findings

- The retained backward queue has depth two. It preloads `k=0`, then enqueues
  `k+1` immediately after dequeuing `k`, before the current Cast/Add. This is the
  useful rolling MTE2/Vector overlap pattern taken from the external
  `flash-linear-attention-npu` study; deeper staged pipelines were not copied
  because this reduction has a serial accumulator dependency and small `m`.
- Forward profiling for `[1024,4096],m=4` showed stable AIV time around
  `10.3-10.4 us`, with MTE3 about `66%`, MTE2 about `21%`, Scalar about `14%`,
  and negligible Vector work. It is an output-copy problem, not a compute
  problem.
- The Cube equivalent would express forward as a degenerate `K=1` broadcast
  product and backward as `[S*D,m] @ ones[m,1]`. For public `m=2/4/8`, Cube
  padding/layout traffic and very low K/N utilization dominate; no Cube
  candidate was implemented.

## Additional CANNJudge results

- Submission `6a8886ec82cffa8f167f907a`: terminal `Pass`, precision 5/5,
  times `[4.58, 21.96, 1392.00, 49.46, 5.02]`, derived mean score `71.5145`.
- Submission `6a888abc82cffa8f167fe54c`: terminal `Pass`, precision 5/5,
  times `[4.66, 21.80, 1392.80, 47.48, 4.26]`, derived mean score `74.0741`.
- Corresponding platform `best_time` values were
  `[2.02, 15.22, 1370.10, 46.66, 2.58]`. The platform does not map these five
  entries to public shapes, so no hidden-case identity is inferred.
- The first submission containing the 32x128 blockDim grid and retained FP16
  `m=2` path reached `Pass` 5/5 with times
  `[4.78,22.38,1380.58,47.44,4.98]` and derived score `71.6476`. Because this
  was below the prior no-grid `74.0741`, the grid was reverted before the next
  platform isolation submission.

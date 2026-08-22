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

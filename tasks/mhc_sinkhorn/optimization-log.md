# MhcSinkhorn optimization log

Only measured experiments belong here.

| Candidate | Hypothesis | Correctness | Score | Decision | Evidence |
|---|---|---:|---:|---|---|
| retained correctness baseline | A complete-matrix FP32 state with one-core matrix ownership is the simplest contract-complete baseline | CPU 86/86; local A3 ACLNN 21/21 | local A3 orthogonal sum 3986.280861 us | keep as reference | CANN 8.5 official 910B package build; local A3 event benchmark and msprof, 2026-08-22 UTC |

## Retained baseline performance model

`agent_loop.py diagnose --task mhc_sinkhorn --name baseline-performance-model`
completed after guard, official 910B build, and local A3 validation passed.
The Harness had no configured profiler evidence at diagnosis time. Its
`tiling,memory,pipeline,scalar,bandwidth,synchronization` tags remain
`static_hypothesis`, not measured findings.

### benchmark_observed

The retained source was measured by CANN ACL runtime events on local
`Ascend910_9382`. Each case has five warmups and nine active samples; every
active sample averages repeated identical ACLNN launches. The full samples are
in `runs/performance/mhc_sinkhorn_retained_baseline.{jsonl,md}`.

- Matrix-count sweep (`N=8`, FP32, iterations=20, no mask): total latency was
  67.5726, 67.7210, 71.2964, 270.5790, and 1068.0640 us for 1, 4, 48, 192,
  and 768 matrices. The corresponding cost per matrix was 67.5726, 16.9303,
  1.48534, 1.40927, and 1.39071 us.
- The large fixed launch/setup cost is therefore real for very small matrix
  counts. After 48 matrices, per-matrix cost improves only 6.37% by 768
  matrices; steady-state is not primarily a task-amortization problem.
- Same-shape iterations sweep (`[192,8,8]`) measured 25.2032, 76.9184,
  270.5790, and 1297.4900 us for 1, 5, 20, and 100 iterations. The fit is
  `T = 12.8365 us + iterations * 12.8480 us`, R-squared 0.999999. At
  iterations=20, the fitted fixed term is about 4.75% of total latency, so the
  recurrent iteration body is the first steady-state target.
- N=4/6/8 measured 69.0972/158.1630/270.5790 us at the same 192-matrix,
  20-iteration regime. This is close to N-squared scaling and contradicts a
  purely fixed-control model across N.
- FP16 and FP32 measured 271.1420 and 270.5790 us. This is expected from the
  shared FP32 recurrent state; storage dtype is not a useful first split.
- Absent/scalar/full masks measured 270.5790/271.5640/271.4700 us. Mask
  staging is below 0.4% here and is not a first optimization target.

### profile_observed

The task-local CANN 8.5 msprof wrapper successfully exported task-based
`PipeUtilization` for 185 identical `[192,8,8]`, FP32, iterations=20 launches.
The full aggregate is in
`runs/performance/mhc_sinkhorn_profile_retained_baseline.md`.

- Median Task Duration: 270.780 us; median AIV time: 268.758 us.
- Exported Block Dim: 48. The export did not provide direct per-core active
  utilization, so blockDim is not relabeled as occupancy.
- Median Scalar pipe time/ratio: 267.049 us / 99.4%.
- Median Vector pipe time/ratio: 0.273 us / 0.1%.
- Median MTE2 time/ratio: 1.741 us / 0.6%; MTE3: 0.488 us / 0.2%.
- Median Task Wait Time was 0 us, with a maximum of 214.98 us. This is only the
  exported task-wait field, not evidence for a specific pipeline stall.

### static_hypothesis and pattern selection

The profile localizes the dominant symptom to scalar work. Source inspection
then identifies the repeated `GetValue`/`SetValue` row and column loops and
V/S event boundaries as the concrete code path responsible for almost the
entire iteration slope. This source-to-profile mapping is a hypothesis for the
next experiment, not an additional measured counter.

The most relevant playbook candidates are now, in order:

1. `vector.scan_blocking` / scalar-control removal: restructure several
   independent matrices so row/column normalization can use Vector work
   instead of scalar element loops while retaining each matrix's serial
   iteration dependency and numerical stage order.
2. `common.regime_autotune`: retain a latency path for matrixCount below the
   useful lane group and a throughput path for steady-state counts. The matrix
   sweep is direct evidence for this regime boundary.
3. `common.working_set_liveness`: keep the grouped FP32 state in UB and avoid
   workspace or GM round trips while changing layout.

`vector.mte_v_overlap` is not first: measured MTE2+MTE3 ratios total under 1%.
Simple queue-overhead batching is also not first on its own; it matters only if
it enables scalar-to-Vector reformulation. Cube, mixed-CV credit windows, and
sparse staging do not match this AIV-only dense tiny-matrix recurrence and are
excluded.

No kernel modification or CANNJudge submission was made while establishing
this initial model.

## Optimization results after the retained baseline

All local timings below are `benchmark_observed` proxy evidence from
`Ascend910_9382`, not formal ascend910b performance. Every promoted kernel
passed the CPU contract suite and the complete 21-case local ACLNN matrix
before benchmarking. Numerical reorder candidates were rejected on any
precision failure regardless of speed.

| Candidate | Main mechanism | Proxy score (us) | Decision | Key evidence |
|---|---|---:|---|---|
| N=8 Vector reduction/broadcast | replace scalar GetValue/SetValue iteration body | 555.452397 | PROMOTE | about 7.18x faster than scalar reference |
| N=4/6 pad-to-8 | fixed physical row width with logical N operations | 398.247602 | PROMOTE | all N/mask/dtype cases passed |
| padded repeat fusion | fuse independent padded rows into repeats | 380.276399 | PROMOTE | N=4/6 improved |
| N=8 matrix batch=4 | amortize tiny-matrix Vector/control work | 253.438997 | PROMOTE | N=8 count192 13.699 us |
| all-N batch=4 | apply batching to padded regimes | 245.811798 | PROMOTE | full matrix passed |
| N=8 pairwise column tree | shorten column dependency chain | 238.035002 | PROMOTE | count768 33.417 us |
| N=8 remove redundant same-pipe barriers | rely on ordered V-pipe dependencies | 236.308000 | PROMOTE | full correctness; about 0.73% proxy gain |
| Newton reciprocal | reciprocal refinement plus multiply | 269.780398 | REJECT | slower |
| Div(1,denom)+Mul | reduce broadcast division form | 237.898202 | REJECT | slower |
| N=8 BlockReduce row max/sum | use one-block reduction primitive | 206.876600 | PROMOTE | 12.45% gain; iter100 39.98 -> 26.74 us task profile |
| N=8 batch=8/16 | enlarge matrix tile | 211.550400 / 207.524599 | REJECT | worse aggregate proxy |
| N=4/6 BlockReduce sums | exploit zero padding for fixed 8-lane sums | noisy aggregate near 207 | PROMOTE | stable N=4/6 improvement of roughly 25–30% |
| padded barrier removal | remove same-pipe barriers | about 17.0/19.1 for N=4/6 | REJECT | worse than retained padded path |
| padded pairwise columns | tree column reduction | negligible or incorrect v1 | REJECT | v1 polluted N=6 padding; corrected v2 only marginal |
| N=8 batch upper bound 32 | allow up to 32 matrices in one local batch | **198.175802** | PROMOTE | current retained proxy; 4.21% beyond submission 4 source |
| plain Reciprocal+Mul | approximate reciprocal normalization | n/a | REJECT | FP32 max error about 1.6e-3 exceeded tolerance |
| TQue -> TBuf I/O | remove queue lifecycle with explicit events | 198.486603 | REJECT | correct but 0.16% slower |
| cross-matrix strided column Div | 8 strided Div calls instead of 32 contiguous calls | 197.821600 aggregate | REJECT | iter100 regressed 27.47 -> 33.15 us |
| column Div mask hoist | set fixed mask once | 201.132600 | REJECT | no gain; count768 regressed |
| full denominator broadcast | 8 Adds plus one continuous Div | 207.559200 | REJECT | broadcast cost dominated |
| iteration partial unroll x2 | reduce runtime loop control | 201.992800 | REJECT | code expansion did not reduce Scalar symptom |
| pooled compute TBuf | reduce InitBuffer calls | 201.361001 | REJECT | no fixed-latency improvement |

The retained source uses complete-matrix core ownership, quotient/remainder
core balance, full-chain FP32 UB residency, compile-time N/mask/dtype regimes,
pad-to-8 for N=4/6, N=8 batching up to 32 matrices, BlockReduce row
reductions, and the pairwise N=8 column tree. It preserves the original
matrix-local iteration order and public ABI.

## Current profiler model

Evidence categories are intentionally separate:

- `profile_observed`: retained N=8, FP32, 192 matrices, iterations=100 had
  Task/AIV/Vector/Scalar times near 27.18/25.41/19.88/8.78 us. Vector active
  was about 78%, while ArithmeticUtilization reported FP32 plus miscellaneous
  ratios around 20.8%. ResourceConflictRatio reported UB bank-group conflict
  about 1.8–2.2%, bank conflict 0, and Vector resource conflict 0.
- `profile_observed`: one matrix, iterations=20 had Task/AIV/Vector/Scalar
  times 4.18/3.664/2.401/1.706 us. MTE2 and MTE3 were only 0.204 and 0.125 us.
  The profiled host/task-wait median was 15.4 us and must not be interpreted as
  kernel pipeline time.
- `benchmark_observed`: the retained optimized orthogonal proxy score is
  198.175802 us, versus 3986.280861 us for the original scalar baseline
  (about 20.1x lower aggregate latency).
- `static_hypothesis`: remaining low arithmetic density is consistent with
  short reduction/division dependency chains and instruction issue gaps, but
  the profiler did not export a direct V-pipe stall reason. It is not evidence
  that MTE/V overlap, UB conflict, queue setup, or loop control is dominant.

The attempted pipeline layouts above falsify the simple versions of those
hypotheses. The retained contiguous-repeat layout is faster than strided
cross-matrix Div, repeated mask setup is not material, and additional
broadcast work is not amortized by fewer Div calls.

## CANNJudge evidence and submission budget

The verified target is contest `6a7bf087a52e0f540a88e167`, public problem
`302`, internal problem `6a7c2267a52e0f540a8a02bd`. Four successful
submissions have been consumed from the user-authorized budget of 20:

| Submission | ID | Precision | Public times | Retained change |
|---:|---|---|---|---|
| 1 | `6a89cac782cffa8f16941bfe` | 1/1 Pass | 10.9 / 15.8 | N=8 batch4 |
| 2 | `6a89cc0782cffa8f16945766` | 1/1 Pass | 11.62 / 15.6 | all-N batch4 |
| 3 | `6a89cd4c82cffa8f169495b9` | 1/1 Pass | 11.42 / 13.24 | N=8 pairwise column tree |
| 4 | `6a89d16582cffa8f16956f3e` | 1/1 Pass | 11.50 / 11.36 | N=8 BlockReduce |

The visible best times at submission 4 were 3.34 / 2.3 us. No fifth
submission is justified yet: the current retained proxy is only 4.21% better
than the source used by submission 4, below the requested roughly 10% batching
threshold for conserving platform attempts. Submission budget used: **4/20**.

## Candidate-tree disposition

- P0 matrix ownership and full iteration residency: already satisfied.
- P0 history side-store: not applicable; the executable ABI has no
  `normOut/sumOut`.
- P1 matrix batching, compile-time N specialization, pad-to-8, and
  broadcast/sync work: implemented and measured as recorded above.
- P1 loop invariants: repeat descriptors and fixed dimensions are already
  compile-time/local setup; mask hoisting and loop unroll did not help.
- P2 N=4 MicroAPI: unavailable on the required ascend910b V220 compiler;
  a compile probe failed because `AscendC::MicroAPI` is not defined.
- P2 reciprocal normalization: exact alternatives were slower; the fast
  approximation failed FP32 precision.
- P2 hot-iteration unroll: factor two was slower, so further code-size growth
  is not supported by evidence.

The large-matrix FlashSinkhorn streaming regime, Cube, mixed-CV credit
windows, and sparse staging remain structurally inapplicable to dense
4/6/8-by-4/6/8 AIV recurrence.

## Harness reliability fix

An ascend910b MicroAPI compile probe exposed that the upstream outer CMake
build could return success after OPC printed a kernel compile failure, leaving
stale objects available to later targets. The task-local `scripts/build.sh`
now captures the binary build log and fails on OPC error signatures even when
the outer process exits zero. A clean official build passed after this guard
was added. This is a build-evidence fix, not a kernel performance candidate.

## Final retained-source verification

The source was rebuilt and reinstalled after all rejected candidates were
reverted. Final evidence on 2026-08-22 UTC:

- repository guard: no task-specific interface rule configured; repository
  guard skipped cleanly, and `git diff --check` passed;
- CPU contract/reference: 86/86 passed;
- local A3 ACLNN: 21/21 passed on `Ascend910_9382`;
- official target: ascend910b Host, FP16 kernel, FP32 kernel, ACLNN library,
  and `.run` package all built successfully;
- local A3 orthogonal proxy rerun: 197.239401 us. The difference from the
  198.175802 us retained measurement is treated as normal proxy variation;
- relative to the 206.876600 us proxy source used by submission 4, the final
  rerun is about 4.66% faster, below the user's roughly 10% submission gate;
- CANNJudge doctor verified contest `6a7bf087a52e0f540a88e167`, public problem
  `302`, and internal problem `6a7c2267a52e0f540a8a02bd` together.

No fifth submission was made. Submission budget remains 4/20 used.

## Candidate: N=8 minimum eight matrices per core

**HYPOTHESIS:** the retained Host always launches every available AIV core once
`matrixCount >= coreCount`, leaving only four matrices per core at the common
192-matrix regime. The N=8 Vector microkernel may benefit if fewer cores each
execute at least eight matrices.

**MECHANISM:** only for N=8 and at least two natural core waves, choose
`usedCoreNum = min(aivCoreCount, ceil(matrixCount / 8))`. Kernel arithmetic,
batch layout, ABI, and iteration order are unchanged.

**EXPECTED ASCEND EFFECT:** improve Vector work density and reduce per-core
setup amortization; accept lower core occupancy only if the denser batch more
than compensates.

**RESOURCE COST:** no extra UB, register, workspace, or code-size cost.

**CORRECTNESS RISK:** quotient/remainder ownership and final partial batches;
no numerical risk.

**EVALUATION:** full CPU/A3 correctness, then the identical orthogonal proxy,
especially N=8 matrixCount 192/768 and iterations 1/5/20/100.

**RESULT — REJECT:** correctness passed (CPU 86/86 and local A3 ACLNN 21/21),
but the orthogonal proxy regressed from 197.239401 to 211.620199 us (about
7.3%). At matrixCount=192, iterations=100 regressed from 27.544001 to
38.909999 us. Lowering core parallelism cost more than the denser per-core
batch recovered, so quotient/remainder distribution across all available
cores is retained.

After reverting, CPU 86/86, local A3 ACLNN 21/21, and the official ascend910b
Host/FP16/FP32/package build all passed again. The same proxy reran at
197.367000 us, confirming restoration within normal measurement variation.

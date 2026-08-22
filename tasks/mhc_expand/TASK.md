# MhcExpand verified competition contract

## Provenance

- **PLATFORM FACT** — CANNJudge display problem ID: `301`.
- **PLATFORM FACT** — CANNJudge internal problem identifier (`_id`): `6a7c1a74a52e0f540a89d39b`.
- **PLATFORM FACT** — Canonical problem name: `mhcexpand`; version: `v1`; title: `【B组简单题】mHC-expand 算子（前向与反向）`.
- **PLATFORM FACT** — Contest: display ID `90`, internal identifier `6a7bf087a52e0f540a88e167`, canonical name `ct_starcup_aiop_g2`, title `中国电信星辰杯高校AI算子开发挑战赛（B组）`.
- **PLATFORM FACT** — The identity above was resolved from the live CANNJudge group/contest/problem APIs. The ZIP filename was not used as identity evidence.
- **PLATFORM FACT** — The current statement was fetched from `GET /api/problems/6a7c1a74a52e0f540a89d39b`; its content-exact LF-normalized snapshot is [problem.md](problem.md), and the platform's unmodified `desc` text has UTF-8 SHA-256 `cf63a12ee4e6ffbdae120c5d2e79396eccf25cffbe783a215d55a6281a9f3a09`.
- **PLATFORM FACT** — The project package was fetched again on 2026-08-19 from `GET /api/problems/6a7c1a74a52e0f540a89d39b/package`; its filename was `MhcExpand_problem_301_template.zip`, size was 3882 bytes, and that downloaded ZIP's SHA-256 was `46602d2eb98c2f616ca537fd15d3ca25197658db5abf590fd5ae3aafc75454c8`.
- **PLATFORM FACT** — Only public problem metadata and the public project package were accessed. Hidden testcase contents were not requested or inspected; the public API exposed only five opaque testcase references.

## Authority and interface

- **PLATFORM FACT** — The freshly downloaded platform statement and platform package are the authoritative evidence for this analysis.
- **PLATFORM FACT** — The current platform package defines operator `MhcExpand` with required input `x`, required output `o`, optional integer attribute `mhc_mult` defaulting to `2`, and optional boolean attribute `backward` with no explicit default in the generated OpDef.
- **PLATFORM FACT** — The current platform package declares `x` and `o` as `bfloat16` or `float16`, format `ND`, and configures `ascend910b`.
- **TEMPLATE FACT** — The checked-in template now defines exactly the same visible interface: `x -> o`, `{BF16, FP16}`, optional `mhc_mult=2`, and optional `backward`.
- **TEMPLATE FACT** — Recursive comparison against the package fetched on 2026-08-19 produced no differences. Both trees have the same seven paths and the same deterministic content-tree SHA-256 `9422a25da48b09d492591398afdafd0ecad2ebdaf32b1173440a373f329b5066`.
- **PLATFORM FACT** — The statement defines disjoint legal logical signatures: rank-2 `[S,D]` is forward and rank-3 `[S,m,D]` is backward. Legal mode selection is therefore uniquely recoverable from input rank without consulting `backward`.
- **PLATFORM FACT** — CANN 8.5 `OpAttrDef` documentation says `Bool()` sets only the attribute type, while `Bool(bool value)` sets a default, and requires the value-taking overload for an `OPTIONAL` attribute. The official template's `.AttrType(OPTIONAL).Bool()` therefore does not explicitly encode a standards-compliant default.
- **TEMPLATE FACT** — As a diagnostic only, running the unchanged fresh package through the locally installed CANN 9.0 op generator produced `attr_backward.defaultValue=false` and `.ATTR(backward, Bool, false)`. This is evidence of the generator's zero-default behavior, but it is not proof of the declared CANN 8.5 evaluator behavior.
- **DESIGN PROPOSAL** — Normalize legal calls by shape: rank 2 selects forward and rank 3 selects backward. Treat `backward=false` as forward and `backward=true` as backward when the attribute is present; if the attribute is unavailable/null, use the rank-derived mode. Reject a present attribute that contradicts the input rank. This preserves the conventional meaning of `backward`, agrees with the observed generated `false` default for forward, and does not depend on the undocumented omission behavior.
- **UNRESOLVED** — The public statement still does not literally state the truth-value mapping, and the CANN 8.5 documentation conflicts with the template's missing bool default. These are platform/interface documentation defects, but they no longer block correctness for the published legal domain because rank determines the operation uniquely.

## Mathematical contract

### Forward

- **PLATFORM FACT** — Forward `x` is the single residual stream, a rank-2 ND tensor of shape `[S, D]`, where `S` is token count and `D` is hidden dimension.
- **PLATFORM FACT** — `S` and `D` are positive integers.
- **PLATFORM FACT** — `mhc_mult = m` is a positive integer expansion factor; the current package default is `m = 2`.
- **PLATFORM FACT** — Forward output `o` has shape `[S, m, D]` and the same dtype as `x`.
- **PLATFORM FACT** — For every `0 <= i < S`, `0 <= k < m`, and `0 <= j < D`, `o[i,k,j] = x[i,j]`.
- **PLATFORM FACT** — Forward is a materialized replication, equivalent to `x.unsqueeze(1).expand(-1, m, -1).clone()`; it is not merely a zero-stride view.

### Backward

- **PLATFORM FACT** — Backward logical input is `o_grad`, a rank-3 ND tensor of shape `[S, m, D]` and dtype BF16 or FP16.
- **PLATFORM FACT** — Backward logical output is `x_grad`, a rank-2 ND tensor of shape `[S, D]` and the same dtype as its input.
- **PLATFORM FACT** — For every `0 <= i < S` and `0 <= j < D`, `x_grad[i,j] = sum(k=0..m-1, o_grad[i,k,j])`.
- **PLATFORM FACT** — The platform package exposes the logical backward tensors through the same physical names `x` and `o`; it does not expose separate `o_grad` or `x_grad` OpDef ports.
- **UNRESOLVED** — The statement requires consistency with the referenced implementation and highlights FP16/BF16 accumulation precision, but it does not explicitly define accumulation order, accumulation dtype, or final rounding beyond the mathematical sum and output dtype.

## Supported domain and boundaries

- **PLATFORM FACT** — Supported tensor dtypes are exactly BF16 and FP16 in both the current statement and current platform package.
- **PLATFORM FACT** — Supported data format is ND.
- **PLATFORM FACT** — The statement's legal-domain wording is `S > 0`, `D > 0`, and `m > 0`; it gives no finite maxima.
- **PLATFORM FACT** — Published coverage includes `(S,D,m)=(64,256,2)`, `(1024,4096,4)`, and `(8192,7168,8)`, plus `S=1`, `D=1`, and non-aligned dimensions.
- **PLATFORM FACT** — Published expansion-factor coverage includes `m in {2,4,8}`.
- **PLATFORM FACT** — The coverage values are examples/test coverage, not an explicit restriction overriding the stated positive-integer domain.
- **UNRESOLVED** — Empty tensors, zero dimensions, negative dimensions, `m=0`, negative `m`, overflow-sized outputs, invalid ranks, and rank/shape disagreement with `mhc_mult` are outside the published legal domain; the required rejection/error behavior is not stated.

## Shape and dtype inference contract

- **PLATFORM FACT** — In forward mode, InferShape must map `[S,D]` to `[S,mhc_mult,D]`.
- **PLATFORM FACT** — In backward mode, InferShape must map `[S,mhc_mult,D]` to `[S,D]`.
- **PLATFORM FACT** — In both modes, InferDataType must copy the input dtype to the output; the legal input/output dtype set is `{BF16, FP16}`.
- **DESIGN PROPOSAL** — InferShape should select the operation from the input rank, then require any present `backward` value to agree (`false` for rank 2, `true` for rank 3). A missing/null `backward` must not be dereferenced and does not prevent exact shape inference.

## Correctness and scoring

- **PLATFORM FACT** — The task requires correctness for both directions, both supported dtypes, legal generalized shapes, single-token/single-dimension boundaries, and non-aligned dimensions.
- **PLATFORM FACT** — CANNJudge's documented defaults for FP16 and BF16 are `rtol=1e-3`, `atol=1e-3`, and error-rate tolerance `tol=1e-3` when a testcase output does not override them.
- **UNRESOLVED** — Public problem metadata does not reveal whether hidden testcase outputs override those defaults; hidden testcase contents were deliberately not accessed.
- **PLATFORM FACT** — Problem metadata is `score_mode=0` (partial-testcase scoring), `use_baseline=false`, difficulty `1`, template type `custom_template`, and CANN version `8.5.0`.
- **PLATFORM FACT** — The contest uses score ranking with default total-score aggregation.
- **PLATFORM FACT** — For each passing testcase, the performance score contribution is `tbest / submission_time * 100`, where `tbest` is the global fastest passing time for that testcase; the single-problem score is the average contribution over passing testcases.
- **PLATFORM FACT** — Only precision-passing testcases record time and contribute to performance score; platform documentation says each testcase runs repeatedly and uses mean time, defaulting to five iterations when not otherwise configured.
- **PLATFORM FACT** — The statement requires bandwidth-oriented forward copy optimization, backward reduction optimization, good shape-dependent partitioning, and generalization, but provides no absolute latency target or time limit.

## Target environment

- **PLATFORM FACT** — CANN version: `8.5.0` from live problem metadata.
- **PLATFORM FACT** — Target compute unit/SOC configuration: `ascend910b` from both the fresh package root CMake and the OpDef `AddConfig`.

## Template comparison

- **TEMPLATE FACT** — Recursive comparison found the same seven file paths and no byte differences between checked-in `workspace/code/` and the fresh package.
- **TEMPLATE FACT** — The shared Host OpDef is `x -> o`, optional `mhc_mult=2`, optional `backward`, and BF16/FP16.
- **TEMPLATE FACT** — The shared InferShape and InferDataType implementations are empty success stubs.
- **TEMPLATE FACT** — The shared TilingData contains only `uint32_t length`; Host Tiling reads `mhc_mult` and `backward` but stores neither and passes only the input element count.
- **TEMPLATE FACT** — The shared kernel has empty `Init` and `Process` bodies.

## Explicit contradiction record

- **UNRESOLVED** — The final block of the live problem description abruptly specifies an unrelated `softmax(src, index=None, ptr=None, num_nodes=None, dim=0)` interface and asks CANNJudge to add missing parameters. It contradicts the MhcExpand title, all preceding semantics, and the fresh MhcExpand package; it is preserved in `problem.md` and excluded from the MhcExpand contract pending platform correction.
- **UNRESOLVED** — The optional `backward` attribute has no explicit source-level default and no stated truth-value mapping in the public statement/package comments. CANN 8.5 documentation requires an explicit optional-attribute default, while a CANN 9.0 generation diagnostic synthesized `false`.

## Implementation gate

- **TEMPLATE FACT** — The correctness baseline is implemented in the single official source tree `workspace/code/`; its persistent build targets remain `ASCEND_COMPUTE_UNIT=ascend910b` and `.AddConfig("ascend910b")`.
- **DESIGN PROPOSAL** — Formal build command (`BUILD_CMD`): `tasks/mhc_expand/scripts/build-platform-910b.sh`.
- **DESIGN PROPOSAL** — Local-only validation command (`LOCAL_VALIDATE_CMD`): `MHC_EXPAND_DEVICE_ID=4 tasks/mhc_expand/scripts/validate-local-a3.sh`. It is not the platform `VALIDATE_CMD` and must not be presented as 910B execution evidence.
- **TEMPLATE FACT** — `build-platform-910b.sh` performs a clean CANN 8.5 + `Ascend-cann-910b-ops` build from `workspace/code/`, verifies the two persistent target fields, and produces Host, FP16/BF16 kernels, ACLNN library, and an `ascend910b` package in an independent temporary build root.
- **TEMPLATE FACT** — `validate-local-a3.sh` copies the same official source into a script-owned temporary mirror, changes only the two build/SOC target fields to the runtime-confirmed `ascend910_93`, rejects any other recursive diff as `A3_ADAPTATION_SCOPE_VIOLATION`, and uses a separate build/install/cache root.
- **TEMPLATE FACT** — On 2026-08-21, the final CANN 8.5 + 910B OPS clean build produced Host tiling, FP16/BF16 kernels, ACLNN library, and an `ascend910b` package: `PLATFORM 910B BUILD PASS`.
- **TEMPLATE FACT** — The final local A3 mirror run used device 4 and passed all 44 tests, including both modes/dtypes, `m in {1,2,3,4,5,8}`, boundary/non-aligned/published shapes, default-attribute behavior, and invalid-contract rejection: `LOCAL A3 CORRECTNESS PASS`. This is local A3 evidence, not `PLATFORM CORRECTNESS PASS`.
- **TEMPLATE FACT** — The retained kernel uses complete-row batching in both directions and a two-slot rolling-prefetch input queue for generic backward `m>1`. It assigns about eight complete-row tasks per core only for `S<192`; multi-tile rows and `S>=192` use `min(taskCount,AIV cores)`, preserving 48 cores for wide/medium public shapes. BF16 and FP16 multipliers other than `m=2` retain serial FP32 accumulation. FP16 backward `m=2` uses one same-dtype Add because exhaustive finite-pair evidence found no numerical difference from the FP32-sum reference; its only bit difference is mathematically equal signed zero. That path uses its actual six-byte-per-element UB footprint instead of reserving unused FP32 buffers. Backward `m=2/4` now use internal compile-time TilingKey variants to eliminate the runtime reduction bound while all other positive multipliers retain the generic fallback. Detailed gate/profile evidence is in `optimization-log.md`.
- **PLATFORM FACT** — Eight measured submissions have reached terminal platform status `Pass`, and all five result entries in every submission passed with `precision_ratio=1`. The first three IDs are `6a88808d82cffa8f167f1b9d`, `6a8886ec82cffa8f167f907a`, and `6a888abc82cffa8f167fe54c`; later exact-once submissions used sanitized reporting that deliberately omitted identifiers.
- **PLATFORM FACT** — The latest platform-successful backward-`m=2/4` compile-time specialization result was `[2.06,22.68,1376.62,47.60,2.32]`, with `best_time=[2.02,15.22,1366.96,46.66,2.32]` and derived mean `92.4979`. It improved the prior retained `90.9719` result and established the current global best `2.32` for the fifth opaque entry. Earlier vectors and candidate decisions remain recorded in `optimization-log.md`.
- **PLATFORM FACT** — Submission responses expose no aggregate score; every documented mean is derived with the platform formula, not returned directly.
- **TEMPLATE FACT** — Harness credential inheritance is now working through the RSA-encrypted workflow. Its submission loop treats `Pass` as terminal and creates no retry submission while polling.

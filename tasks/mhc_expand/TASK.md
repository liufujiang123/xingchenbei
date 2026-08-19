# MhcExpand verified competition contract

## Provenance

- **PLATFORM FACT** — CANNJudge display problem ID: `301`.
- **PLATFORM FACT** — CANNJudge internal problem identifier (`_id`): `6a7c1a74a52e0f540a89d39b`.
- **PLATFORM FACT** — Canonical problem name: `mhcexpand`; version: `v1`; title: `【B组简单题】mHC-expand 算子（前向与反向）`.
- **PLATFORM FACT** — Contest: display ID `90`, internal identifier `6a7bf087a52e0f540a88e167`, canonical name `ct_starcup_aiop_g2`, title `中国电信星辰杯高校AI算子开发挑战赛（B组）`.
- **PLATFORM FACT** — The identity above was resolved from the live CANNJudge group/contest/problem APIs. The ZIP filename was not used as identity evidence.
- **PLATFORM FACT** — The current statement was fetched from `GET /api/problems/6a7c1a74a52e0f540a89d39b`; its content-exact LF-normalized snapshot is [problem.md](problem.md), and the platform's unmodified `desc` text has UTF-8 SHA-256 `cf63a12ee4e6ffbdae120c5d2e79396eccf25cffbe783a215d55a6281a9f3a09`.
- **PLATFORM FACT** — The current project package was fetched from `GET /api/problems/6a7c1a74a52e0f540a89d39b/package`; its filename was `MhcExpand_problem_301_template.zip`, size was 3882 bytes, and SHA-256 was `db79ed5d977e16e817c9595522676f09659db9fe8305a538b358892a3d7d0986`.
- **PLATFORM FACT** — Only public problem metadata and the public project package were accessed. Hidden testcase contents were not requested or inspected; the public API exposed only five opaque testcase references.

## Authority and interface

- **PLATFORM FACT** — The freshly downloaded platform statement and platform package are the authoritative evidence for this analysis.
- **PLATFORM FACT** — The current platform package defines operator `MhcExpand` with required input `x`, required output `o`, optional integer attribute `mhc_mult` defaulting to `2`, and optional boolean attribute `backward` with no explicit default in the generated OpDef.
- **PLATFORM FACT** — The current platform package declares `x` and `o` as `bfloat16` or `float16`, format `ND`, and configures `ascend910b`.
- **TEMPLATE FACT** — The checked-in template defines required input `x`, required output `y`, required integer attribute `expand_num`, and FP16/FP32 dtypes.
- **TEMPLATE FACT** — The checked-in visible OpDef is therefore stale relative to the current platform package; the platform package explicitly proves that the active submission interface has changed.
- **UNRESOLVED** — The public statement describes forward and backward as separate logical signatures but does not state explicitly which value of package attribute `backward` selects each direction or what omission of that optional attribute means.

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
- **UNRESOLVED** — Exact InferShape mode selection remains tied to the undocumented `backward` boolean mapping/default.

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

- **TEMPLATE FACT** — Recursive comparison found the same seven file paths in the checked-in tree and fresh package.
- **TEMPLATE FACT** — The trees are not byte-for-byte equal; several CMake/header differences are newline-only.
- **TEMPLATE FACT** — Material differences exist in `op_host/mhc_expand.cpp`, `op_kernel/mhc_expand.cpp`, and `op_kernel/tiling_key_mhc_expand.h`: port name `y` became `o`; `expand_num` became `mhc_mult`; `backward` was added; FP32 was replaced by BF16; and kernel parameter names/template dtype selections changed accordingly.
- **TEMPLATE FACT** — Both checked-in and fresh package InferShape/InferDataType implementations are empty success stubs.
- **TEMPLATE FACT** — Both checked-in and fresh package TilingData contain only `uint32_t length`; fresh Host Tiling stores only input element count and does not pass shape, expansion factor, or direction.
- **TEMPLATE FACT** — Both kernels have empty `Init` and `Process` bodies.

## Explicit contradiction record

- **UNRESOLVED** — The final block of the live problem description abruptly specifies an unrelated `softmax(src, index=None, ptr=None, num_nodes=None, dim=0)` interface and asks CANNJudge to add missing parameters. It contradicts the MhcExpand title, all preceding semantics, and the fresh MhcExpand package; it is preserved in `problem.md` and excluded from the MhcExpand contract pending platform correction.
- **UNRESOLVED** — The optional `backward` attribute has no explicit default and no stated truth-value mapping in the public statement/package comments.

## Implementation gate

- **UNRESOLVED** — Status: **BLOCKED** for correctness implementation until the platform confirms the exact `backward` truth-value mapping and omitted-attribute behavior.
- **DESIGN PROPOSAL** — After that single interface ambiguity is resolved, start from the fresh package interface, not the stale checked-in OpDef, and preserve the verified public contract exactly.

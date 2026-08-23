---
name: xingchen-kernel-optimizer
description: Ascend operator engineering router for competition tasks: reuse context, protect contract/correctness, diagnose Vector/Cube/Mixed-CV bottlenecks, discover research-derived optimization experience, route to the right tool, and validate evidence-backed candidates.
---

# Xingchen Kernel Optimizer

Use this as the compact professional map for Ascend operator work. Keep specialized Ascend knowledge and tool routing here; let Codex handle ordinary C++/Python engineering, code reading, refactoring, and general debugging without a scripted ritual.

## Start/resume without rereading

At a new Codex conversation for a task:

```bash
python3 tools/context_state.py bootstrap --task <task> --new-session
python3 tools/task_state.py --task <task>
```

Do this once. Reuse stable instructions already in model context. If context may be stale:

```bash
python3 tools/context_state.py check --task <task>
python3 tools/context_state.py use --task <task> <path>
```

Unchanged -> reuse context. Changed -> consume diff first. Full reread is last resort. Load another Skill through `context_state.py use` so it is not ritualistically reopened.

## Contract, correctness, and operator development

Authority: current competition/platform evidence -> repository/task contract -> official template -> official Ascend Skills/references -> assumptions.

Protect public ABI and mathematical semantics. Keep scheduling choices internal to Host Tiling, workspace, templates, and kernel implementation. Do not narrow supported dtype/shape/mode ranges, guess hidden cases, weaken correctness, or report unexecuted gates as passing.

For a new/uncertain operator resolve only correctness-critical facts first: ABI, equations/dependencies, supported domain, precision semantics, target CANN/SOC, required files/package. Then build the simplest contract-complete baseline.

When available, use design experience as a catalog rather than a mandatory ceremony:

```bash
python3 tools/ascend_design_analyze.py --task <task>
python3 tools/ascend_design_analyze.py --task <task> --select-pattern <pattern-id>
```

Read catalog summaries once; Codex chooses relevant patterns. Design topics include contract consistency, stage/dependency graph, task ownership, layout, Host Tiling/tails, memory lifetime/workspace, precision, reduction/scan/recurrent/sparse/Cube/Mixed-CV architecture, target separation, and validation boundaries.

Official Ascend Skills are on-demand specialists: `ascendc-operator-design`, `code-gen`, `compile-debug`, `precision-debug`, `mssanitizer`, `performance-eval`, `performance-optim`.

## Ascend performance resource model

Before optimizing, identify the hot-path class and actual dependency graph.

- **Vector:** `S -> MTE2 -> UB -> V -> UB -> MTE3`. Inspect copy/compute/copy overlap, UB working set, reduction/scan passes, scalar issue overhead, core occupancy.
- **Cube:** `S -> MTE2 -> L1 -> MTE1 -> L0A/L0B -> M -> L0C -> FIX`. Inspect M/N/K ownership, L1 residency, streaming side, K-loop pipeline, output-block bubbles, Cube occupancy.
- **Mixed C/V:** `AIC/Cube <-> workspace/on-chip handoff <-> AIV/Vector`, plus MTE and cross-core sync. Distinguish true-ready dependencies from workspace-slot reuse; model legal producer lead, live workspace, ring capacity, prologue/drain.

Never copy fixed tile sizes, stage counts, ring depths, sync intervals, AIC:AIV ratios, or thresholds from another kernel.

## Bottleneck diagnosis

Evidence confidence is:

`profile-observed > configured hypothesis > static source risk`

Static source patterns suggest risks; they do not prove measured bottlenecks.

Bottleneck tags:
- `pipeline`: bubbles/stalls between overlap-capable stages.
- `memory` / `bandwidth`: excessive GM transfers, MTE waits, repeated scans/materialization.
- `cache`: poor L1/L2 reuse, thrashing, bad working-set/order.
- `compute`: Vector/Cube useful execution dominates.
- `underutilization`: too few or imbalanced useful tasks.
- `scalar`: control/address/div-mod/launch/sync issue dominates short kernels.
- `synchronization`: flags/barriers/handshakes serialize work.
- `tiling`: tile/regime choice hurts occupancy/resources/tails.
- `sparse`: gather/page/index traffic and fragmentation dominate.

Preferred integrated diagnosis when supported:

```bash
python3 tools/agent_loop.py diagnose --task <task> --name pre-candidate
```

Direct source/profile diagnosis:

```bash
python3 tools/ascend_perf_analyze.py \
  --task <task> \
  [--profile-file <file>] \
  [--operator-class auto|vector|cube|mixed_cv] \
  [--bottleneck-hint <tag>] \
  [--advanced]
```

Use source-only analysis as hypothesis generation. If the next decision depends on whether pipeline/cache/sync behavior is real, profile it. Profile wrappers may emit genuine `HARNESS_OPERATOR_CLASS`, `HARNESS_BOTTLENECKS`, and `HARNESS_PROFILE_NOTE` markers.

## Research-derived optimization experience

When present, `config/ascend_optimization_patterns.json` is the reusable experience registry. Keep this summary catalog in mind; load detailed `when / try / avoid / evidence` only when relevant.

**Common**
- `common.multicore_balance` — rebalance legal independent work across cores.
- `common.dependency_aware_partition` — keep recurrence chains local; parallelize orthogonal axes.
- `common.working_set_liveness` — derive buffers/rings from live ranges, async hazards, cache/on-chip fit.
- `common.locality_schedule` — order independent work for reuse and GM bandwidth deconfliction.
- `common.fuse_fragmented_staging` — assemble sparse/paged/indexed fragments directly in UB/L1 when legal.
- `common.remove_scalar_overhead` — hoist/simplify address, control, tiny-loop, and sync issue work.
- `common.regime_autotune` — benchmark a small hardware-pruned regime set.
- `common.hardware_path_audit` *(advanced)* — inspect anomalously slow dtype/shape/API lowering.

**Vector**
- `vector.mte_v_overlap` — overlap MTE2 / V / MTE3 across independent tiles.
- `vector.reduce_pass_fusion` — reduce repeated full GM scans with a numerically valid online/fused state.
- `vector.scan_blocking` — keep serial scan state local and vectorize orthogonal work.
- `vector.instruction_fusion` — replace scalar/slice issue with whole-tile/compound Vector work.
- `vector.microapi_register_kernel` *(advanced)* — RegTensor/MaskReg only for a proven hotspot.

**Cube**
- `cube.preload_pipeline` — address output-block boundary bubbles beyond inner-loop ping-pong.
- `cube.asymmetric_residency` — keep the reusable operand resident; stream the other side.
- `cube.split_axis_for_occupancy` — split reduction work only when occupancy gain beats extra reduction/write cost.

**Mixed C/V**
- `mixed_cv.credit_window` — replace unnecessary lockstep with bounded dependency-safe producer lead.
- `mixed_cv.sync_batch` — batch legal cross-core synchronization with correct tail flush.
- `mixed_cv.workspace_liveness_ring` — size workspace rings from lifetime/dependency/cache fit.
- `mixed_cv.internal_fusion` *(advanced)* — shorten proven handoff/materialization only with API/SOC support.

## From diagnosis to candidate

If class and bottleneck are already known:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class <vector|cube|mixed_cv> \
  --bottleneck <tag>
```

Use additional `--bottleneck` only for supported symptoms. Use `--advanced` only after ordinary decomposition/layout/movement/pipeline fixes are insufficient and target API/SOC evidence justifies it.

Prefer diagnosis-generated ranking when available because it preserves evidence provenance, but Codex may choose another catalog pattern when the real dependency/resource model supports it.

When mechanisms interact, usually inspect in this order:
1. dependency-safe task decomposition / occupancy;
2. layout and data movement;
3. working-set residency / buffering / pipeline;
4. synchronization/window;
5. tiling/regime tuning;
6. advanced hardware-path/register microkernel.

## Evidence freshness

Runtime evidence is valid only for the world state that produced it. When `tools/evidence_fingerprint.py` is present, Harness runs bind each `build / validate / bench / profile / platform` stage to:

- evaluated implementation (`subject_hash`);
- task/gate configuration and referenced scripts;
- validation/benchmark case set;
- selected non-secret CANN/SOC/device environment identity;
- stage command context.

Use:

```bash
python3 tools/task_state.py --task <task>
```

before citing old evidence. `fresh` means the current implementation and relevant execution context still match. `stale` means rerun the affected gate before claiming it as proof. Legacy evidence without a fingerprint is `unknown`, never silently treated as fresh.

A source change makes old correctness/build evidence stale, but source changes are expected between performance candidates and therefore do **not** by themselves make two benchmark scores incomparable. Benchmark comparison requires the same `bench_context_hash`: build/bench gate, cases, environment and relevant config must match. If the context differs, start a new best-score lineage instead of claiming faster/slower.

Optional task config can narrow/extend fingerprint scope with `EVIDENCE_SUBJECT_PATHS`, `EVIDENCE_CASE_PATHS`, `EVIDENCE_CONTEXT_PATHS`, and non-secret `EVIDENCE_ENV_KEYS`. Never put credentials in these fields.

## Candidate evidence loop

For each meaningful candidate record only: hypothesis + evidence level, expected Ascend resource effect, one major mechanism, added UB/L1/workspace/code-size and correctness/precision risk, and exact same-case evaluation.

Run target build -> correctness -> same-case benchmark. Profile only when it answers the next concrete question. Decide `PROMOTE`, `REJECT`, or `INCONCLUSIVE`; keep failed experiments as evidence and do not stack unproven mechanisms.

Promotion requires interface compatibility, required correctness/precision coverage, **fresh** evidence for the current subject, measured improvement beyond noise, and no legal-domain narrowing. Local proxy results remain proxy; platform conclusions require actual platform evidence.

## Tool routing

- already loaded context -> `context_state.py`; do not reread.
- current task/evidence freshness -> `task_state.py`.
- inspect the current fingerprint -> `evidence_fingerprint.py`.
- design experience -> `ascend_design_analyze.py`.
- unknown performance bottleneck -> `agent_loop.py diagnose` or `ascend_perf_analyze.py`.
- known class+bottleneck -> `ascend_perf_plan.py`.
- real pipeline/cache/sync evidence -> configured profile / `ascendc-operator-performance-eval`.
- compile/runtime failure -> `ascendc-operator-compile-debug`.
- numerical mismatch -> `ascendc-operator-precision-debug`.
- target-specific optimization question -> `ascendc-operator-performance-optim`.
- platform identity/package/submission -> CANNJudge tooling; submission is explicit only.

If a referenced Harness tool is absent on the current branch, do not invent results or casually rebuild it. Use the corresponding official Ascend Skill/manual evidence path, or deliberately port the generic Harness capability.

## Reporting

Do not spend tokens proving Harness compliance. Report new contract facts/conflicts, diagnosed bottleneck and evidence level, selected/rejected experience, meaningful code decisions, gate results with freshness, measured performance with comparability, and blockers.

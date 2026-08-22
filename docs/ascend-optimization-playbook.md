# Ascend-first kernel optimization playbook

This playbook is the generic performance layer of the competition harness. It complements the official Ascend performance skills; it does not replace them.

The target loop is:

`contract -> correctness baseline -> source/profile diagnosis -> operator class -> bottleneck tags -> research-derived shortlist -> one candidate -> build/correctness/bench -> keep/reject`

## 1. Evidence hierarchy

Performance reasoning must keep three evidence levels separate:

1. **profile-observed** — profiler output, explicit profile markers, measured utilization/stall/cache symptoms;
2. **configured hypothesis** — a task/operator hint supplied because the profiler cannot expose a needed metric;
3. **static source risk** — source constructs that make a bottleneck plausible.

A static source risk is never a measured bottleneck. It is only a reason to profile or test a candidate.

The harness records these separately in `record["diagnosis"]`.

## 2. Classify the hot path

Every performance pass resolves one primary class:

- `vector`: typical flow `GM -> UB -> V -> UB -> GM`;
- `cube`: typical flow `GM -> L1 -> L0A/L0B -> Cube -> L0C/FIX -> GM`;
- `mixed_cv`: substantial Cube and Vector stages exchange staged tiles/workspace.

Classification precedence is:

`configured PERF_OPERATOR_CLASS -> HARNESS_OPERATOR_CLASS profile marker -> static source inference`

Use `PERF_OPERATOR_CLASS=auto` by default. Override only when the generic scanner cannot see generated/lowered code or the source intentionally contains inactive implementations for multiple classes.

## 3. Run diagnosis through the harness

Normal pre-candidate diagnosis:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --name pre-candidate
```

`diagnose` performs the configured guard/build/correctness gates and, when `PROFILE_CMD` exists, runs the profiler before generating the shortlist.

For a cheap static-only pass after fresh correctness evidence:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --skip-build \
  --skip-validate \
  --skip-profile \
  --name source-scan
```

The existing profile mode also attaches diagnosis automatically:

```bash
python3 tools/agent_loop.py profile \
  --task <task> \
  --name <question>
```

The task-local JSON record contains:

- resolved operator class and its source;
- `observed_bottlenecks`;
- configured hints;
- `static_risk_tags`;
- profile/source evidence snippets;
- conflicts between profile and source classification;
- a small ranked list from `config/ascend_optimization_patterns.json`.

## 4. Optional stable profile markers

Profiler wrappers may emit these lines:

```text
HARNESS_OPERATOR_CLASS=vector
HARNESS_BOTTLENECKS=pipeline,memory,bandwidth
HARNESS_PROFILE_NOTE=MTE2 waits dominate the steady state
```

Allowed classes are `vector`, `cube`, `mixed_cv`.

Useful bottleneck tags are:

`pipeline`, `memory`, `bandwidth`, `cache`, `compute`, `latency`, `underutilization`, `scalar`, `synchronization`, `tiling`, `sparse`.

Markers are optional. The analyzer also conservatively recognizes textual symptoms such as pipeline bubbles, sync stalls, cache misses and low core utilization. Do not print a tag from a wrapper unless the underlying profiler actually supports the claim.

## 5. Generic task configuration

Task configs may add:

```bash
# auto is recommended.
PERF_OPERATOR_CLASS=auto

# Optional comma/semicolon-separated source roots.
# Defaults to WORKSPACE_DIR plus TASK_DIR.
PERF_SOURCE_DIRS='tasks/<task>/workspace/code'

# Optional hypothesis tags when a profiler cannot expose the symptom directly.
PERF_BOTTLENECK_HINTS=''

# Keep shortlists small.
PERF_PLAN_LIMIT=5

# Advanced patterns stay off by default.
PERF_ADVANCED=0
```

These are generic knobs. They must not encode visible testcase IDs, hidden-test guesses, fixed competition scores, or another repository's magic stage/tile/ring values.

## 6. Resource/dependency model

Before editing code, inspect the diagnosis and still reason about the actual dependency graph.

### Vector

Relevant resources typically include:

`Scalar -> MTE2 -> UB -> V -> UB -> MTE3`

Ask whether independent tiles can achieve:

`CopyIn(n+1) || Compute(n) || CopyOut(n-1)`.

Only buffers participating in asynchronous producer/consumer access need multi-buffering. V-only temporaries do not gain overlap merely because they have two copies.

### Cube

Typical resources include:

`MTE2 -> L1 -> MTE1 -> L0A/L0B -> M -> L0C -> FIX`

Ping-pong inside a K loop can still leave bubbles between output blocks. Model prologue, steady state and drain; cross-block preload is a separate candidate from ordinary double buffering.

When one operand is reused and fits L1, model resident and streaming sides asymmetrically instead of spending identical stage counts on both.

### Mixed Cube + Vector

Treat C/V as a bounded producer-consumer system, not automatically as lockstep.

Investigate:

- whether adjacent C/V tasks are independent;
- whether each wait is a true dependency or only workspace-slot reuse;
- legal producer lead distance;
- long-lived versus short-lived workspace values;
- whether a fixed ring is larger than the useful live window/cache working set;
- whether synchronization may be batched without delaying a real dependency.

A credit window, ring depth, sync interval or AIC:AIV ratio must be derived for the current kernel. Never copy the value from a reference repository.

## 7. Research-derived scan

For every candidate cycle ask:

- **dependency axes** — which axis is a true recurrence and which orthogonal axes are independent?
- **working-set liveness** — which values are live simultaneously and which are asynchronously accessed?
- **locality/conflict** — can grouped/swizzled order improve reuse, or can traversal phases be offset to reduce synchronized GM conflicts?
- **movement** — can sparse/paged/reformatted fragments be assembled directly in UB/L1 instead of round-tripping through GM?
- **algorithmic passes** — can related full reduction scans be fused with a numerically valid online state?
- **hardware path** — does the supported dtype/shape actually lower to the intended hardware path?
- **autotune regime** — do different shape/dtype/layout regimes justify a small hardware-pruned candidate set?

The pattern registry records the reusable mechanism and its guard, not reference-project parameter values.

## 8. Candidate generation

Standalone manual lookup remains available:

```bash
python3 tools/ascend_perf_plan.py \
  --task <task> \
  --operator-class <vector|cube|mixed_cv> \
  --bottleneck <tag>
```

However, prefer the shortlist embedded by `agent_loop.py diagnose/profile`, because it preserves the provenance of the tags used for ranking.

Advanced/SOC- or API-sensitive patterns are hidden by default. Expose them only with:

```bash
python3 tools/agent_loop.py diagnose \
  --task <task> \
  --advanced-diagnosis
```

Examples include MicroAPI register microkernels and closer C/V handoff. They should be considered only after ordinary layout/movement/pipeline fixes and target CANN/SOC support are established.

## 9. One-candidate rule

Each candidate must state:

- hypothesis;
- evidence level and bottleneck tag;
- expected resource/pipeline effect;
- one major mechanism changed;
- UB/L1/workspace/code-size cost;
- correctness/precision risk;
- exact same-case evaluation plan.

Do not stack several unproven mechanisms.

Recommended order when mechanisms interact:

1. dependency-safe task decomposition;
2. layout/data movement;
3. buffering/residency/pipeline;
4. synchronization/window tuning;
5. tiling/regime autotune;
6. advanced hardware path/register microkernel.

Recompute memory budgets whenever live buffers change.

## 10. Promotion

A candidate is retained only when:

- public interface is unchanged;
- target build passes;
- required correctness/precision passes;
- same-case performance improves beyond noise;
- required shape/dtype/mode coverage is not narrowed;
- proxy hardware evidence is labeled as proxy;
- CANNJudge conclusions come only from actual platform responses.

Record rejected and inconclusive experiments too. Failure is useful evidence for the next diagnosis.

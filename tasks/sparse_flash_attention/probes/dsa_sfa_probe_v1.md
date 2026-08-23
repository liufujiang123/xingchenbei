# DsaSfa evaluator probe v1

This candidate is an intentionally falsifiable interpretation of the undocumented official `DsaSfa` tensor semantics. It is **not** treated as a platform fact.

## Authoritative baseline

The source layout, filenames, OpDef, dtype constraints, attribute declaration, compute-unit setting, and CMake packaging are based on the freshly verified CANNJudge problem-285 official package. The mature harness branch is used only for tooling and workflow infrastructure; its earlier rewritten `SparseFlashAttention` interface is not authoritative and has been removed from this task branch.

## Hypothesis H1

The official four-input template is interpreted as the final sparse-attention aggregation stage:

- `values`: per-batch value table, final axis `D`;
- `sparse_index`: local KV row indices, final axis `K`;
- `score`: precomputed attention logits, final axis `K`;
- `gate`: positive multiplicative prior/mask with the same flattened layout as `score`;
- `scale`: multiplies `score` before softmax;
- `aggregated`: stable-softmax weighted sum of sparse-gathered `values`;
- `agg_weights`: normalized sparse weights, same shape as `score`.

`score` may contain more rows than `sparse_index`. When the ratio is integral, the probe treats it as query-head broadcasting caused by `KV_N=1`.

Conceptual layout:

```text
values       [B, KV_S, 1, D]
sparse_index [B, Q_S, 1, K]
score/gate   [B, Q_S, Q_N, K]
aggregated   [B, Q_S, Q_N, D]
agg_weights  [B, Q_S, Q_N, K]
```

## Numerical implementation

For each score row:

```text
valid(k) = 0 <= sparse_index[k] < KV_rows_per_batch and gate[k] > 0
logit(k) = score[k] * scale
u(k) = gate[k] * exp(logit(k) - max_valid_logit)
w(k) = u(k) / sum_valid_u
aggregated[d] = sum_k w(k) * values[sparse_index[k], d]
agg_weights[k] = w(k)
```

The first probe uses scalar GM access and an approximate scalar exp. That is deliberate: evaluator feedback should confirm the layout/semantics before investing in vector softmax, UB buffering, or gather optimization.

If runtime shapes do not satisfy H1, the kernel uses an in-bounds deterministic fallback rather than indexing an invented layout.

## Evaluator interpretation

| Result | Interpretation |
|---|---|
| Compile Error | No semantic conclusion; fix only compilation. |
| Runtime/shape error | Strong evidence that H1 shape/dtype inference or launch layout is wrong. |
| Wrong Answer with successful execution | H1 is structurally runnable but one or more semantics are wrong. |
| Some test points Accepted | Strong evidence that part of H1 matches the hidden contract. |
| TLE after successful execution | Semantic hypothesis may still be plausible; this probe is intentionally slow. |
| All Accepted | Freeze semantics, replace scalar implementation with optimized Ascend Vector/Cube dataflow. |

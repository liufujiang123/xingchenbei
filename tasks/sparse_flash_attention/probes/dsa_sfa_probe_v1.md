# DsaSfa evaluator probe v1

This candidate is an intentionally falsifiable interpretation of the undocumented official `DsaSfa` tensor semantics. It is **not** treated as a platform fact.

## Hypothesis H1

The official four-input template is interpreted as the final sparse-attention aggregation stage:

- `values`: per-batch value table, final axis `D`;
- `sparse_index`: local KV row indices, final axis `K`;
- `score`: precomputed attention logits, final axis `K`;
- `gate`: positive multiplicative prior/mask with the same flattened layout as `score`;
- `scale`: multiplies `score` before softmax;
- `aggregated`: stable-softmax weighted sum of sparse-gathered `values`;
- `agg_weights`: normalized sparse weights, same shape as `score`.

`score` may contain more rows than `sparse_index`. When the ratio is integral, the probe treats it as query-head broadcasting caused by `KV_N=1`. For the common conceptual layout:

```text
values       [B, KV_S, 1, D]
sparse_index [B, Q_S, 1, K]
score/gate   [B, Q_S, Q_N, K]
aggregated   [B, Q_S, Q_N, D]
agg_weights  [B, Q_S, Q_N, K]
```

one sparse-index row is shared by all `Q_N` query heads.

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

The first probe uses scalar GM access and a small polynomial/range-reduced approximation for `exp`. That is deliberate: evaluator feedback should confirm the layout/semantics before investing in vector softmax, UB buffering, or gather optimization.

If runtime shapes do not satisfy H1, the kernel uses an in-bounds deterministic fallback (`values` prefix -> `aggregated`, `score` prefix -> `agg_weights`) rather than indexing an invented layout.

## Evaluator interpretation

| Result | What it tells us |
|---|---|
| Compile Error | No semantic conclusion. Fix Ascend C/Host compilation first. |
| Runtime/shape error | Strong evidence that H1 shape/dtype inference or launch layout is wrong. |
| Wrong Answer with successful execution | H1 is structurally runnable; tensor semantics, mask/gate rule, scaling, index broadcast, output meaning, or numerical rule is still wrong. |
| Some test points Accepted | Strong evidence that at least part of H1 matches the hidden contract. Compare accepted vs rejected case metadata/status only; do not attempt to access hidden testcase contents. |
| TLE only after successful execution | H1 may still be semantically plausible; scalar probe is intentionally slow. Correctness evidence should be separated from timing evidence if CANNJudge exposes both. |
| All Accepted | Replace scalar implementation with a correctness-preserving Ascend vector/Cube implementation and begin measured optimization. |

## What to report after submission

Please preserve the exact evaluator feedback available to contestants:

- compile log if compilation fails;
- per-case public status (`Accepted`, `Wrong Answer`, `Runtime Error`, `TLE`, etc.);
- precision ratio if exposed;
- score/time if exposed;
- any public shape/dtype validation error.

Do not change the official OpDef signature between probes unless the platform explicitly says it is allowed.

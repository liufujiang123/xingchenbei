# SparseFlashAttention task contract

Status: contest ABI resolved from the checked-in competition template; correctness baseline implementation is in progress.

## Authoritative sources

For this task use, in order:

1. the current CANNJudge contest instance and the checked-in contest template under `workspace/code/`;
2. `B组困难题_SparseFlashAttention 算子.md` for mathematical semantics;
3. repository `AGENTS.md`;
4. official Ascend C skills and public implementation references.

The previously investigated `DsaSfa` package is not the template for this contest instance and must not be used to redefine this task.

## Contest-visible operator interface

Registered operator: `SparseFlashAttention`

Kernel entrypoint: `sparse_flash_attention`

Required/optional inputs, in the exact template order:

| Pos | Name | Presence | Template dtype | Logical shape |
|---:|---|---|---|---|
| 0 | `query` | required | float16 / float32 | `(B, Q_S, Q_N, 512)` |
| 1 | `key` | required | float16 / float32 | `(B, KV_S, 1, 512)` |
| 2 | `value` | required | float16 / float32 | `(B, KV_S, 1, 512)` |
| 3 | `sparse_indices` | required | int32 | `(B, Q_S, 1, sparse_size)` |
| 4 | `actual_seq_lengths_query` | optional | int32 | `(B,)` |
| 5 | `actual_seq_lengths_kv` | optional | int32 | `(B,)` |
| 6 | `query_rope` | required | float16 / float32 | `(B, Q_S, Q_N, 64)` |
| 7 | `key_rope` | required | float16 / float32 | `(B, KV_S, 1, 64)` |

Outputs, in exact template order:

| Pos | Name | Presence | Template dtype | Logical shape |
|---:|---|---|---|---|
| 0 | `attention_out` | required | float16 / float32 | `(B, Q_S, Q_N, 512)` |
| 1 | `softmax_max_out` | optional | float32 | `(B, 1, Q_S, Q_N)` |
| 2 | `softmax_sum_out` | optional | float32 | `(B, 1, Q_S, Q_N)` |

Attributes, in exact template order:

| Pos | Name | Template default | Required semantics |
|---:|---|---|---|
| 0 | `scale_value` | `0.0884` | multiply the complete content+RoPE score; task statement says scale is processed at float16 precision |
| 1 | `sparse_block_size` | `1` | A2/910B supports powers of two in `[1,128]` |
| 2 | `sparse_mode` | `3` | `0` = no causal mask, `3` = right-down causal |
| 3 | `attention_mode` | `2` | only MLA-absorb mode `2` is supported |
| 4 | `return_softmax_lse` | false | controls auxiliary softmax max/sum outputs |

Do not change these names, positions, optionality, defaults, filenames, operator registration, or kernel entrypoint unless the contest platform explicitly changes the template.

## Logical dimensions

- `B`: batch size.
- `Q_S`: physical query sequence length.
- `KV_S`: physical key/value sequence length.
- `Q_N`: query head count.
- `KV_N = 1`: all query heads share one KV head.
- content head dimension: exactly `512`.
- RoPE dimension: exactly `64`.
- `sparse_size > 0`: selected KV positions per query row.
- layout: BSND / ND contiguous storage.

For the 910B/A2 target named by the template, the statement gives query-head counts `1,2,4,8,16,32,64,128` and sparse block sizes that are powers of two in `[1,128]`.

## Required mathematical computation

For every valid `(batch, query_position, query_head)` row:

1. Read the row `sparse_indices[b, q, 0, :]`.
2. Keep indices that are inside the effective KV length and satisfy the selected mask.
3. Gather content key/value rows and key-RoPE rows from the single shared KV head.
4. Compute MLA-absorb score:

```text
score_k = (
    dot(query[b,q,h,:], key[b,k,0,:])
  + dot(query_rope[b,q,h,:], key_rope[b,k,0,:])
) * scale_value
```

The content dot width is `512`; the RoPE dot width is `64`.

5. Compute stable softmax over the effective sparse positions.
6. Return:

```text
attention_out[b,q,h,:] = sum_k softmax(score)_k * value[b,k,0,:]
```

7. When `return_softmax_lse` is true, also return per-row:

```text
softmax_max_out = max(score)
softmax_sum_out = sum(exp(score - max(score)))
```

The functional NumPy example in the problem statement intentionally omits the RoPE contribution and is therefore not a complete reference implementation.

## Actual sequence lengths

- `actual_seq_lengths_query=None` means `Q_S` for every batch.
- `actual_seq_lengths_kv=None` means `KV_S` for every batch.
- supplied tensors are int32 `(B,)`.
- positions beyond the effective lengths are padding and do not participate.
- sparse indices are valid only when `0 <= index < actual_kv_length`.

The current baseline writes zero attention output for padded query rows. Evaluator evidence should be used to confirm whether auxiliary outputs for padded/empty rows need a different convention.

## Mask semantics

`sparse_mode=0`: no additional causal masking.

`sparse_mode=3`: right-down causal. The current baseline uses the standard right-aligned predicate for effective lengths:

```text
key_position <= query_position + actual_kv_length - actual_query_length
```

This is the interpretation to validate against CANNJudge.

## Sparse block size

The input shape still contains one sparse-index row per physical query token. The correctness baseline consumes the row associated with the current query position. It does not perform an additional remapping for `sparse_block_size`; the expectation is that the supplied `sparse_indices` already encode the shared block selection decision. This must be validated with block-wise evaluator cases.

## Dtype contract

The statement describes float16/bfloat16 logical data, while the contest template currently declares float16/float32. The submission must preserve the contest template declaration. The current baseline therefore implements the template's float16/float32 paths and does not silently add BF16 to the public interface.

`query`, `key`, and `value` are required to share dtype. The baseline permits the two RoPE tensors to independently be float16 or float32 as allowed by the template.

## Current baseline implementation

The first implementation is intentionally correctness-oriented:

- one logical `(B,Q,H)` output row is an independent multicore task;
- sparse K/V and RoPE rows are read directly from GM;
- score dot products accumulate in float32;
- stable online softmax maintains a running max and denominator;
- each AIV core owns a reusable `512 * float32` workspace accumulator for the row it is currently processing;
- workspace size is `usedCoreNum * 512 * sizeof(float)` and does not grow with sequence length or sparse size;
- the final normalized accumulator is cast once into `attention_out`;
- scalar exponential approximation with range reduction;
- no Cube batching, UB gather aggregation, or performance tuning yet.

The implementation is not considered validated until it successfully builds under the contest-compatible CANN toolchain and passes the actual correctness evaluator.

## Remaining correctness questions for evaluator evidence

The public statement/template do not completely specify:

- sentinel values beyond the documented bounds check;
- duplicate/unsorted sparse-index behavior;
- exact expected outputs for a row with no valid/unmasked key;
- auxiliary-output values for padded or empty rows;
- whether block-wise mode requires any additional row remapping beyond the provided `sparse_indices` tensor;
- numerical tolerance and exceptional-value policy;
- whether all declared float32 paths are exercised by the contest.

These uncertainties must not be resolved by changing the public ABI. Use build/runtime/CANNJudge evidence to refine only internal behavior.

## Development order

1. compile the scalar baseline with the contest-compatible CANN toolchain;
2. fix compile/API issues without changing the mathematical design;
3. run local smoke/reference tests where available;
4. submit only when explicitly authorized;
5. use evaluator results to fix semantic or precision failures;
6. only after correctness passes, replace scalar GM-heavy stages with UB/Vector/Cube optimized dataflow.

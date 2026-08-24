# SparseFlashAttention task contract

Status: verified problem identity and submission transport; CANN 8.5 compilation and local A3 correctness pass; permissive Host Tiling now crosses CANNJudge GetWorkspace and reaches Kernel result comparison.

## Authoritative sources

For this task use, in order:

1. the current CANNJudge contest instance and the checked-in contest template under `workspace/code/`;
2. `B组困难题_SparseFlashAttention 算子.md` for mathematical semantics;
3. repository `AGENTS.md`;
4. official Ascend C skills and public implementation references.

The previously investigated `DsaSfa` package is not the template for this contest instance and must not be used to redefine this task.

Verified CANNJudge provenance on 2026-08-23:

- contest ID: `6a7bf087a52e0f540a88e167`;
- public problem ID: `303`;
- internal problem ID: `6a7c22d6a52e0f540a8a098d`;
- live problem name: `sparseflashattention`;
- fresh package filename: `SparseFlashAttention_problem_303_template.zip`;
- fresh package SHA-256: `b87832774789b2d1f9c23a4418d688d43a7085a4cb030572c74c23dd9012ef9a`.

The fresh package preserves the same names, ordering, optionality, defaults, target, and FP16/FP32 OpDef as the original checked-in skeleton. The checked-in Host/Kernel/Tiling files intentionally differ because they contain the implementation.

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

The live statement describes float16/bfloat16, while the fresh official package declares float16/float32. To cover both authoritative artifacts without removing the package's FP32 path, the implementation exposes a deliberate FP16/FP32/BF16 compatibility union at the public OpDef and TilingKey boundary. Local CANN 8.5 evidence proves that all three selectors build and that BF16 reaches successful device execution; whether the platform evaluator accepts the extra BF16 signature remains UNRESOLVED until a platform submission exercises it.

## Current baseline implementation

The first implementation is intentionally correctness-oriented:

- one logical `(B,Q,H)` output row is an independent multicore task;
- sparse K/V and RoPE rows are read directly from GM;
- score dot products accumulate in float32;
- stable online softmax maintains a running max and denominator;
- each AIV core owns a reusable `512 * float32` UB accumulator for the row it is currently processing;
- Host Tiling advertises zero workspace because the kernel keeps the accumulator in per-core UB;
- the final normalized accumulator is cast once into `attention_out`;
- scalar exponential approximation with range reduction;
- scalar GM outputs are explicitly cache-cleaned before row completion;
- auxiliary-output mode uses one AIV core to avoid cross-core cache-line false sharing;
- no Cube batching, UB gather aggregation, or performance tuning yet.

The implementation builds under CANN 8.5 for `ascend910b` with FP16, FP32, and BF16 binaries. A temporary target-only `ascend910_93` mirror passed 11/11 deterministic local ACLNN/NumPy cases, including a true-launch BF16 case, and a 41/41 GetWorkspaceSize-only matrix covering all three same-dtype combinations. Submission `6a8a841282cffa8f16ab684b` reached Kernel result comparison on all three public cases before this compatibility-union revision, proving the prior `561002` came from custom Host over-validation rather than the Kernel mathematics.

## Remaining correctness questions for evaluator evidence

The public statement/template do not completely specify:

- sentinel values beyond the documented bounds check;
- duplicate/unsorted sparse-index behavior;
- exact expected outputs for a row with no valid/unmasked key;
- auxiliary-output values for padded or empty rows;
- whether block-wise mode requires any additional row remapping beyond the provided `sparse_indices` tensor;
- numerical tolerance and exceptional-value policy;
- whether all declared float32 paths are exercised by the contest.
- which individual retired Host validation predicate rejected the platform call; this is no longer correctness-blocking because those checks were not required by the official template;
- whether the evaluator accepts the FP16/FP32/BF16 compatibility union or requires one exact two-dtype signature.

These uncertainties must not be resolved by changing the public ABI. Use build/runtime/CANNJudge evidence to refine only internal behavior.

## Development order

1. compile the scalar baseline with the contest-compatible CANN toolchain;
2. fix compile/API issues without changing the mathematical design;
3. run local smoke/reference tests where available;
4. submit only when explicitly authorized;
5. use evaluator results to fix semantic or precision failures;
6. only after correctness passes, replace scalar GM-heavy stages with UB/Vector/Cube optimized dataflow.

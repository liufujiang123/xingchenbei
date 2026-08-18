# SparseFlashAttention task contract

Status: logical contract extracted; platform ABI reconciliation remains blocked by the unresolved items listed below.

## Authoritative sources and precedence

1. Logical semantics: `tasks/sparse_flash_attention/B组困难题_SparseFlashAttention 算子.md` (the file confirmed by the user as the authoritative `problem.md`).
2. Platform-visible scaffold: every file under `tasks/sparse_flash_attention/workspace/code/`.

The statement defines required behavior. The template defines the visible interface that must remain compatible until wrapper, packing, generated-registration, build, or evaluator evidence proves how the logical API maps to it. Similar names are not mapping evidence.

## Objective

Implement the statement's SparseFlashAttention computation for long-sequence inference: gather sparse key/value positions, compute scaled MLA-absorb attention with the supplied RoPE components and selected mask, and return the required outputs. Preserve the official template ABI while its relationship to the logical interface is being resolved.

## Logical dimensions and layout

| Symbol | Meaning | Contract |
|---|---|---|
| `B` | batch size | Not otherwise bounded in the statement |
| `Q_S` | query sequence length | Not otherwise bounded in the statement |
| `KV_S` | key/value sequence length | Not otherwise bounded in the statement |
| `Q_N` | query head count | Platform-dependent domain below |
| `KV_N` | key/value head count | Exactly `1`; all query heads share one KV head |
| `Q_D`, `KV_D` | content head dimension | Exactly `512` |
| `Dr` | RoPE dimension | Exactly `64` |
| `sparse_size` | indices supplied per query token | Greater than `0`; no upper bound is stated |

The logical tensor layout is `BSND`, `(B, S, N, D)`. The statement also requires ND storage format.

## Logical inputs

The ordering below is the statement's presentation order. It is not asserted to be the platform ABI order.

| Logical input | Required | Shape | Dtype | Statement-defined semantics |
|---|---:|---|---|---|
| `query` | yes | `(B, Q_S, Q_N, 512)` | float16 or bfloat16 | Content query; non-empty and contiguous |
| `key` | yes | `(B, KV_S, 1, 512)` | float16 or bfloat16 | Content key; non-empty and contiguous |
| `value` | yes | `(B, KV_S, 1, 512)` | float16 or bfloat16 | Value; same shape as `key` |
| `sparseIndices` | yes | `(B, Q_S, 1, sparse_size)` | int32 | KV-position indices; each row places valid entries before invalid entries |
| `actual_seq_lengths_query` | no | `(B,)` | int32 | Effective query length per batch; `None` means `Q_S` |
| `actual_seq_lengths_kv` | no | `(B,)` | int32 | Effective KV length per batch; `None` means `KV_S` |
| `queryRope` | yes | `(B, Q_S, Q_N, 64)` | float16 or bfloat16 | Already position-encoded query RoPE component; not rotated inside this operator; non-empty |
| `keyRope` | yes | `(B, KV_S, 1, 64)` | float16 or bfloat16 | Already position-encoded key RoPE component; not rotated inside this operator; non-empty |

`query`, `key`, and `value` must have the same dtype. The statement does not explicitly define whether the RoPE tensors must match that dtype, although each RoPE input independently permits float16 or bfloat16.

## Logical attributes

| Logical attribute | Required | Type/interface representation | Contract |
|---|---:|---|---|
| `scaleValue` | yes | Interface passes `double`; statement says it is processed with float16 precision | Multiplies the complete content-plus-RoPE score and corresponds to `1/sqrt(d_k)`, where `d_k=Q_D=512` |
| `sparseBlockSize` | yes | int64 | `1` is token-wise; values greater than `1` are block-wise and share a selection decision within a block |
| `sparseMode` | yes | int64 | Only documented values are `0` (no mask) and `3` (right-down causal) |
| `attentionMode` | yes | int64 | Only `2` (MLA-absorb) is supported |
| `returnSoftmaxLse` | table says yes | bool | Controls whether both auxiliary softmax outputs are returned; description gives default `False` |

The statement also mentions `pre_tokens` and `next_tokens`, each fixed at the default maximum `2^63-1`, and says contestants need not modify them. It does not establish whether they remain caller-visible attributes or are fixed by a wrapper.

## Logical outputs

| Logical output | Presence | Shape | Dtype | Statement-defined semantics |
|---|---|---|---|---|
| `attentionOut` | required | `(B, Q_S, Q_N, 512)` | float16 or bfloat16 | Sparse attention result; same shape as `query` |
| `softmaxMaxOut` | returned when `returnSoftmaxLse=True` | `(B, 1, Q_S, Q_N)` | float | Per-row softmax maximum |
| `softmaxSumOut` | returned when `returnSoftmaxLse=True` | `(B, 1, Q_S, Q_N)` | float | Per-row sum of `exp(score-max)` |

The statement says the two auxiliary outputs are absent when `returnSoftmaxLse=False`; it does not define the ABI representation of absent outputs.

## Required computation

For each batch, valid query position, and query head:

1. Use the corresponding `sparseIndices` row to gather content keys `K_tilde`, values `V_tilde`, and key RoPE rows `keyRope_tilde` from the single shared KV head.
2. Form the MLA-absorb score by concatenating content and RoPE features, equivalently:

   ```text
   score = (query @ K_tilde^T + queryRope @ keyRope_tilde^T) * scaleValue
   ```

   The content dot product has dimension `512`; the RoPE dot product has dimension `64`; their concatenated feature width is `576`. `value` has only the `512`-wide content component.
3. Apply the selected `sparseMode` mask.
4. Compute stable softmax by taking a row maximum, exponentiating `score-max`, summing those exponentials, and normalizing.
5. Compute `attentionOut = normalized_weights @ V_tilde`.
6. When requested, return the row maximum and exponential sum through the two auxiliary outputs.

The functional example deliberately omits the RoPE contribution and therefore is not a complete reference for the required MLA-absorb computation.

## Sparse-index semantics

- Sparse selection is defined per `(batch, query token, KV head)` row; `KV_N=1`, so every query head uses the row associated with the shared KV head.
- Entries select positions along the KV sequence.
- Each row must contain valid entries first and invalid entries afterward.
- `sparse_size` must be positive.
- The functional example treats indices satisfying `0 <= index < actual_kv_length` as valid and filters all others.
- `sparseBlockSize=1` selects independently per token; block-wise values share selection decisions within a block.

The statement does not specify the invalid sentinel values, duplicate-index behavior, whether valid indices must be sorted, the exact block-wise row-to-block mapping, or the result when a row has no valid and unmasked indices.

## Actual sequence-length semantics

- Both actual-length inputs are optional int32 tensors of shape `(B,)`.
- `None` means the corresponding physical sequence length (`Q_S` or `KV_S`).
- Positions beyond the actual length are padding and do not participate in computation.
- Sparse-index validity in the functional example is checked against the per-batch actual KV length.

The statement does not define output values at padded query positions, permitted actual-length ranges (including zero), or the exact interaction between actual lengths and right-down-causal alignment.

## Attention and mask modes

- Only `attentionMode=2`, MLA-absorb, is supported.
- `sparseMode=0` applies no mask.
- `sparseMode=3` applies a lower-triangular mask with the query sequence right-aligned to the key sequence.

The statement does not give an explicit index predicate for `sparseMode=3`, especially when physical and actual query/KV lengths differ.

## Platform-dependent logical domain

| Platform family named by the statement | `Q_N` | `sparseBlockSize` |
|---|---|---|
| Ascend 950PR/950DT | Any integer from `1` through `128` | Only `1` |
| Atlas A2/A3 | One of `1, 2, 4, 8, 16, 32, 64, 128` | A power of two in `[1,128]` |

The statement covers inference and graph mode. It says the task uses the ACLNN V1 interface and does not include `sinks`.

## Numerical correctness requirements

The statement requires correct results across query-head counts, variable-length sequences, sparse block sizes, and both documented mask modes. It defines stable-softmax intermediates, but provides no numerical tolerance, error metric, NaN/Inf policy, rounding rule beyond `scaleValue` being processed at float16 precision, or authoritative executable reference covering RoPE and masks.

## Performance and scoring

The statement asks contestants to optimize:

- aggregation of discontinuous sparse-gather transfers;
- parallelism of the `Q @ K_tilde^T` computation;
- partitioning across query sequence, query heads, and sparse-index dimensions for multi-batch and variable-length cases.

It provides no scored case table, concrete `B/Q_S/KV_S/sparse_size` cases, scoring formula, metric, direction, case weights, threshold, benchmark command, or profiler contract. No performance claim or shape specialization can be justified until those are supplied.

## Platform-visible official-template ABI

These are template facts, not a claimed mapping to the logical tensors:

| Position | Visible name | Requirement | Declared dtype(s) | Format |
|---:|---|---|---|---|
| input 0 | `values` | required | float16, float32 | ND |
| input 1 | `sparse_index` | required | int32, int64 | ND |
| input 2 | `gate` | required | float16, float32 | ND |
| input 3 | `score` | required | float16, float32 | ND |
| output 0 | `aggregated` | required | float16, float32 | ND |
| output 1 | `agg_weights` | required | float16, float32 | ND |
| attribute 0 | `scale` | optional, default `1.0` | float | N/A |

Additional visible ABI facts:

- Registered operator class/name: `DsaSfa`.
- Kernel entrypoint: `dsa_sfa(values, sparse_index, gate, score, aggregated, agg_weights, workspace, tiling)`.
- Tiling templates specialize only `values` as float32 or float16.
- Host tiling currently passes only `values.GetShapeSize()` as a uint32 `length` field, launches all reported AIV cores, and requests zero workspace.
- Shape and dtype inference callbacks return success without assigning output metadata.
- CMake and registration select `ascend910b`.

This visible interface must remain unchanged until platform evidence establishes that changing it is permitted.

## Unresolved contract/template reconciliation

| ID | Contradiction or ambiguity | Evidence required to resolve it |
|---|---|---|
| ABI-01 | Logical name `SparseFlashAttention` versus registered `DsaSfa` and kernel `dsa_sfa` | Official submission instructions and generated ACLNN symbol/header showing the externally invoked operator |
| ABI-02 | Eight logical tensor inputs (six required, two optional) versus four required template tensor inputs | Wrapper/packing source, generated registration JSON/code, or evaluator call signature and tensor construction |
| ABI-03 | No unique mapping exists from `query/key/value/queryRope/keyRope/actual lengths` to `values/gate/score`; names alone are insufficient | Wrapper source plus packing layout, shapes, strides, offsets, and ownership/lifetime rules |
| ABI-04 | Logical `sparseIndices` resembles template `sparse_index`, but logical dtype is int32 while the template also accepts int64 | Evaluator call signature and generated type constraints; representative runtime descriptors |
| ABI-05 | Five logical attributes plus fixed `pre_tokens/next_tokens` versus one template attribute | Wrapper/generated attribute binding and authoritative defaults/constant insertion |
| ABI-06 | Required `scaleValue` passed as double and processed as float16 versus optional float `scale=1.0` | Generated ACLNN declaration, wrapper conversion code, and evaluator values |
| ABI-07 | One required plus two conditional logical outputs versus two required template outputs | Wrapper unpacking/packing contract and generated output descriptors for both `returnSoftmaxLse` values |
| ABI-08 | Logical output names/roles do not establish mappings to `aggregated` and `agg_weights` | Wrapper source or evaluator assertions for each visible output |
| ABI-09 | Logical float16/bfloat16 tensors versus template float16/float32 tensors and no bfloat16 tiling key | Target-platform template revision or wrapper conversion/packing evidence |
| ABI-10 | Auxiliary logical outputs are float, while template outputs independently admit float16 or float32 | Generated dtype inference rules and runtime output descriptors |
| ABI-11 | Logical shapes are explicit, while template inputs have no shape constraints and inference callbacks do not set output shape/dtype | Generated metadata, build artifacts, wrapper allocation logic, or evaluator-provided output tensors |
| ABI-12 | Optional actual-length inputs have no visible template slots | Wrapper packing/constant/default behavior and evaluator cases with `None` and non-`None` lengths |
| ABI-13 | Required RoPE inputs have no visible template slots | Packing/wrapper layout and evaluator cases with nonzero RoPE contributions |
| ABI-14 | `sparseBlockSize`, `sparseMode`, `attentionMode`, and `returnSoftmaxLse` have no visible template attributes | Wrapper specialization, packed metadata, tiling data source, or generated ABI evidence |
| ABI-15 | The statement covers 950PR/950DT and Atlas A2/A3 domains, while the template selects `ascend910b` | Competition target hardware/SOC list, build flags, and evaluator machine identity |
| SEM-01 | Invalid indices must trail valid indices, but sentinel values are not specified; the example only demonstrates bounds filtering | Official reference implementation or evaluator cases containing each permitted sentinel/out-of-range form |
| SEM-02 | Duplicate and unsorted valid indices are not specified | Reference behavior and dedicated evaluator cases |
| SEM-03 | Output for a row with no valid and unmasked key is not specified | Reference/evaluator case defining attention and auxiliary outputs for an empty effective row |
| SEM-04 | Block-wise selection is described as shared within a block, but the mapping from `Q_S` rows to block decisions is not defined | Formal `sparseBlockSize` indexing rule and block-tail examples |
| SEM-05 | Right-down-causal behavior lacks an exact predicate for unequal physical/actual query and KV lengths | Reference formula and evaluator examples covering unequal lengths and padding |
| SEM-06 | Padding does not participate, but output contents at padded query positions are unspecified | Reference/evaluator expected outputs for padded query rows |
| SEM-07 | The attribute table calls `returnSoftmaxLse` required while also assigning default `False` | Authoritative ACLNN signature/default behavior |
| SEM-08 | Exact auxiliary max/sum values under masking, empty rows, and padding are not fully defined | Reference implementation and evaluator assertions for both auxiliary tensors |
| SEM-09 | `scaleValue` corresponds to `1/sqrt(512)`, but the allowed value domain and whether arbitrary caller values must be honored are not stated | Evaluator inputs and platform API documentation |
| SEM-10 | Valid ranges for actual sequence lengths, including zero, are not stated | Platform validation rules and boundary evaluator cases |
| SEM-11 | The statement requires matching dtypes for `query/key/value`, but does not state whether `queryRope/keyRope` must match them or whether `attentionOut` must match `query` | Generated API constraints and mixed-dtype evaluator cases or an explicit prohibition |
| SEM-12 | No bounds are given for `B`, `Q_S`, `KV_S`, or `sparse_size` beyond `sparse_size>0`; zero-length physical sequences are not addressed | Official supported-shape manifest and boundary cases |
| SEM-13 | Non-empty/contiguous constraints are explicit for some inputs but not individually stated for every logical tensor | Generated API validation rules and evaluator cases for value, RoPE, indices, and actual-length tensors |
| NUM-01 | No correctness tolerance, comparison metric, exceptional-value policy, or complete RoPE/mask reference is supplied | Official validation command, reference implementation, input generator, and tolerance policy |
| PERF-01 | No scored cases or scoring metric are supplied | Official benchmark command, case manifest, score parser, direction, and promotion rule |

## Allowed internal implementation changes

After the ABI is resolved, internal Host Tiling, tiling data, workspace, kernel organization, memory planning, sparse gather, matrix multiplication, stable softmax, multicore partition, and pipeline strategy may be changed without changing the resolved platform-visible contract.

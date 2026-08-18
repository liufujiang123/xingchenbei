# SparseFlashAttention design

Status: architecture reconciliation and correctness-baseline proposal only. No implementation or performance candidate is authorized until the ABI and semantic blockers in `TASK.md` are resolved.

## 1. Evidence boundary

This document separates:

- **Logical contract facts**, taken from `B组困难题_SparseFlashAttention 算子.md`, confirmed by the user as the authoritative problem statement.
- **Visible ABI facts**, taken from the complete official template under `workspace/code/`.
- **Proposals**, explicitly labeled as such and not treated as competition facts.

The logical names cannot be assigned to similarly named template parameters without wrapper, packing, generated-registration, build, or evaluator evidence.

## 2. Reconciled architecture view

The available evidence exposes two layers with an unresolved boundary:

```text
Logical ACLNN V1 formulation from the statement
  query, key, value, sparseIndices,
  actual query/KV lengths, queryRope, keyRope,
  scaleValue, sparseBlockSize, sparseMode,
  attentionMode, returnSoftmaxLse
                         |
                         | unresolved wrapper / packing /
                         | generated registration / ABI layer
                         v
Official template ABI
  DsaSfa(values, sparse_index, gate, score;
         scale=1.0)
    -> (aggregated, agg_weights)
                         |
                         v
Host tiling -> tiling key -> dsa_sfa kernel
```

The middle layer is not present in the repository. Consequently, the diagram records a required relationship but does not assert how any tensor crosses it.

## 3. Logical semantics

For each valid batch/query/head row, the statement requires sparse gather followed by MLA-absorb attention:

```text
K_tilde       = gather(key, sparseIndices)
V_tilde       = gather(value, sparseIndices)
keyRope_tilde = gather(keyRope, sparseIndices)

score = (query @ K_tilde^T
         + queryRope @ keyRope_tilde^T) * scaleValue
score = apply sparseMode mask to score

rowMax       = max(score)
expScore     = exp(score - rowMax)
rowSum       = sum(expScore)
attentionOut = (expScore / rowSum) @ V_tilde
```

The content dot-product width is `512`, the RoPE dot-product width is `64`, and value accumulation produces `512` output features. The single KV head is shared by all query heads. Full shapes, dtypes, modes, and unresolved semantic boundaries are recorded in `TASK.md`.

## 4. Current official-template architecture

### 4.1 Build and packaging

- The top-level CMake project finds the ASC package, selects `ascend910b`, creates a run package named `custom`, and adds `op_host` and `op_kernel`.
- The host CMake file generates operator/ACLNN sources from the host source, builds `cust_optiling` and `cust_opapi`, and packages both libraries.
- The kernel CMake file builds `ascendc_kernels`, links it to the tiling library, and adds it to the package.

### 4.2 Host registration and tiling

The visible host implementation:

1. Registers `DsaSfa` with four required tensors, two required outputs, and optional float attribute `scale=1.0`.
2. Registers only the `ascend910b` configuration.
3. Reads AIV core count and UB size from the platform.
4. Fetches all four visible input tensors and the `scale` attribute.
5. Uses only input-0 (`values`) dtype to select a float16/float32 tiling template.
6. Stores only the total element count of `values` in `DsaSfaTilingData.length` as uint32.
7. Launches the reported AIV core count and requests zero workspace.
8. Returns success from shape and dtype inference without assigning output metadata.

The template does not derive the logical dimensions, actual lengths, sparse size, masks, block size, RoPE layout, optional-output state, or a logical-to-packed mapping.

### 4.3 Kernel

The visible `dsa_sfa` kernel receives the four inputs, two outputs, workspace, and tiling pointers. It reads the single `length` field, calls an empty `Init`, and then an empty `Process`. It therefore provides no current correctness baseline or additional mapping evidence.

## 5. Logical-to-visible mapping

No row below is treated as resolved merely because names look related.

| Logical item | Visible candidate(s) | Status | Evidence needed |
|---|---|---|---|
| `query` | `values`, `gate`, or `score` | unresolved | Wrapper/packing code and runtime shapes/strides |
| `key` | `values`, `gate`, or `score` | unresolved | Wrapper/packing code and runtime shapes/strides |
| `value` | `values` is lexically suggestive only | unresolved | Wrapper/packing code and runtime shapes/strides |
| `sparseIndices` | `sparse_index` is lexically and partly type-compatible | unresolved | Evaluator call and generated type/shape metadata |
| `actual_seq_lengths_query` | no visible slot | unresolved | Wrapper default/packing representation |
| `actual_seq_lengths_kv` | no visible slot | unresolved | Wrapper default/packing representation |
| `queryRope` | no visible slot | unresolved | Wrapper/packing representation |
| `keyRope` | no visible slot | unresolved | Wrapper/packing representation |
| `scaleValue` | visible `scale` | unresolved; requiredness, type, precision, name, and default differ | Generated ACLNN signature and wrapper conversion |
| `sparseBlockSize` | no visible attribute | unresolved | Wrapper specialization or packed metadata |
| `sparseMode` | no visible attribute | unresolved | Wrapper specialization or packed metadata |
| `attentionMode` | no visible attribute | unresolved | Wrapper specialization or packed metadata |
| `returnSoftmaxLse` | no visible attribute | unresolved | Wrapper specialization and output-allocation behavior |
| `pre_tokens`, `next_tokens` | no visible attributes | unresolved | Generated signature or proof that the wrapper fixes both constants |
| `attentionOut` | `aggregated` is lexically suggestive only | unresolved | Evaluator output binding and expected shape/dtype |
| `softmaxMaxOut` | `agg_weights` or packed output | unresolved | Wrapper output layout and evaluator assertions |
| `softmaxSumOut` | `agg_weights` or packed output | unresolved | Wrapper output layout and evaluator assertions |

## 6. Simplest correctness-baseline proposal

This section is a proposal, not an implemented or validated design.

### 6.1 Entry gate

Before code generation:

1. Resolve every ABI mapping needed to access all required logical inputs, attributes, and outputs without changing the visible interface.
2. Resolve the semantic cases that would otherwise require invented behavior: invalid/empty sparse rows, block-wise index mapping, right-down-causal predicate, padded query outputs, and optional-output representation.
3. Obtain the official build and correctness commands, reference behavior, and tolerance.
4. Configure and snapshot the immutable interface guard.

If the evidence shows that the supplied template is for a different operator or that a different official template must be used, follow the platform evidence rather than adapting `DsaSfa` speculatively.

### 6.2 Proposed baseline decomposition

After the gate is satisfied, use independent logical rows `(batch, query position, query head)` as the initial work units. Each row shares the batch's single KV head and its `sparseIndices` row. Distribute rows across available cores with a simple balanced assignment; process the sparse-index dimension in bounded tiles inside each row.

This decomposition is proposed because every output row depends only on its query/query-RoPE row, the selected rows of the shared KV tensors, the relevant actual lengths, and the selected mask. No claim is made yet about the optimal core axis or tile size.

### 6.3 Proposed correctness-first row algorithm

For one resolved logical row:

```text
load query content and query RoPE
determine effective query/KV lengths and whether the query row participates

pass 1 over the sparse-index row:
    stop/skip according to the resolved invalid-index rule
    apply the resolved right-down-causal predicate when selected
    gather key content and key RoPE
    compute content_dot + rope_dot
    apply the required scale conversion and scale the score
    update a numerically stable row maximum

pass 2 over the same effective indices:
    recompute the scaled score
    accumulate exp(score - row_max) into row_sum
    accumulate exp(score - row_max) * gathered_value into the output vector

normalize the output vector by row_sum
cast/store attentionOut in the required output dtype
conditionally store row_max and row_sum through the resolved auxiliary-output ABI
```

This two-pass proposal favors transparent correctness and bounded intermediate storage. It deliberately accepts repeated gather/dot work for the baseline; changing it to online softmax, materialized scores, Cube-oriented batched matrix multiplication, or fused gather/matmul is optimization work and must wait for a passing baseline.

### 6.4 Proposed precision policy

Use float32 dot-product accumulation, softmax maximum/sum, exponential values, normalization, and value accumulation for the initial float16/bfloat16 baseline, then cast `attentionOut` to the required dtype. Preserve the statement's explicit rule that the caller's `scaleValue` is processed with float16 precision before multiplying the complete content-plus-RoPE score.

This is a baseline design choice intended to reduce numerical risk; the official evaluator tolerance and exceptional-value policy are still required before it can be declared sufficient.

### 6.5 Proposed Host Tiling responsibilities

Once the ABI is known, Host Tiling should derive from resolved runtime inputs rather than case IDs:

- logical `B`, `Q_S`, `KV_S`, `Q_N`, `sparse_size`, dtype, and target SOC;
- presence/default representation of actual query and KV lengths;
- `sparseBlockSize`, `sparseMode`, `attentionMode`, and optional-output state;
- a balanced row-task count and sparse tile size subject to the device-reported memory resources;
- workspace and internal tiling metadata required by the selected baseline path.

No concrete tiling-data layout, block count, tile length, workspace size, or UB allocation can be finalized from the current ABI evidence.

### 6.6 Proposed symbolic memory plan

The initial row path would require bounded local buffers for:

| Proposed buffer | Logical contents |
|---|---|
| query content | one `512`-element query vector |
| query RoPE | one `64`-element query-RoPE vector |
| sparse index tile | a bounded slice of one int32 index row |
| gathered key content | selected `512`-element key rows, tiled as required |
| gathered key RoPE | selected `64`-element key-RoPE rows, tiled as required |
| gathered value | selected `512`-element value rows, tiled as required |
| score/exp temporaries | one sparse tile or scalar stream in float32 |
| output accumulator | one `512`-element float32 vector |
| softmax state | float32 row maximum and sum |

Buffer counts, alignment, double buffering, and exact byte totals remain intentionally unspecified until the target SOC, ABI, compiler path, and device resource query are confirmed.

### 6.7 Baseline coverage required before optimization

The correctness evaluator must cover, within the confirmed target platform's stated domain:

- float16 and bfloat16 logical inputs;
- supported query-head counts;
- token-wise and supported block-wise selection;
- `sparseMode=0` and `sparseMode=3`;
- `None` and nontrivial actual query/KV lengths;
- nonzero query/key RoPE contributions;
- valid-prefix plus invalid-suffix sparse rows;
- both values of `returnSoftmaxLse`;
- boundary/tail shapes supplied by the platform.

Cases for duplicates, empty effective rows, zero actual lengths, and padded query outputs cannot be assigned expected results until the corresponding semantics are supplied.

## 7. Precision and correctness risks

| Risk | Why it matters under the statement | Required resolution/control |
|---|---|---|
| Scale conversion | Interface value is double but computation uses float16 precision | Match official conversion/rounding evidence exactly |
| Content-plus-RoPE score | Omitting the `64`-wide RoPE dot product violates MLA-absorb semantics | Include nonzero-RoPE reference cases |
| Stable softmax | Long or high-magnitude rows can overflow without max subtraction | Retain the defined max/subtract/exp/sum flow |
| Mask ordering | Masked or invalid entries must not enter max, sum, or value accumulation | Obtain exact invalid and causal predicates |
| Variable lengths | Padding is excluded from computation but padded output values are undefined | Obtain reference behavior before storing padded rows |
| Empty effective rows | `max` and division are undefined without a specified convention | Do not choose zero/NaN/other behavior without evidence |
| Auxiliary outputs | Their exact behavior depends on scale, mask, padding, and optional-output ABI | Validate them independently against evaluator output |
| Logical/template dtype mismatch | BF16 is required logically but absent from template declarations | Resolve wrapper conversion or obtain corrected template |

## 8. Unresolved contradictions and required evidence

The exhaustive reconciliation register is the table `Unresolved contract/template reconciliation` in `TASK.md`. Its unresolved items are:

- ABI-01 through ABI-15: operator identity, tensor arity/mapping, attribute binding, output binding, dtype/shape inference, optional inputs/outputs, RoPE/mode transport, and target SOC.
- SEM-01 through SEM-13: invalid/duplicate/empty sparse rows, block-wise indexing, causal alignment, padding, optional-output default, auxiliary values, scale domain, actual-length/shape bounds, dtype relationships, and per-input contiguity requirements.
- NUM-01: numerical reference and tolerance.
- PERF-01: scored cases and scoring/evaluation contract.

None of these is resolved by the empty kernel or by lexical similarity between logical and template names.

## 9. Deferred optimization candidates

Only after the proposed baseline builds and passes the official correctness evaluator should individual measured candidates be considered:

- sparse gather aggregation and locality;
- row versus head versus sparse-index multicore partitioning;
- Matmul/MMAD utilization for content and RoPE score components;
- sparse tile sizing and tails;
- GM/L1/UB residency and buffering;
- Vector/Cube overlap;
- online softmax or reduced score materialization.

Each candidate must preserve the resolved ABI and full required domain, change one major dimension, and pass guard, build, validation, and official benchmarking before promotion.

## 10. Implementation readiness checklist

- [x] Authoritative logical statement identified and extracted.
- [x] Complete official template inspected.
- [x] Logical-to-visible mapping recorded without assumptions.
- [x] Contradictions and required evidence catalogued.
- [x] Correctness-baseline algorithm proposed without implementation.
- [ ] External operator symbol and wrapper/packing ABI resolved.
- [ ] All required logical inputs/attributes/outputs mapped.
- [ ] Invalid, empty-row, block-wise, causal, and padding semantics resolved.
- [ ] Numerical reference and tolerance supplied.
- [ ] Target SOC and supported platform subset confirmed.
- [ ] Build, validation, benchmark, and profile commands configured.
- [ ] Interface guard configured and snapshotted.
- [ ] Baseline implemented, built, and validated.

The design is not ready for code generation while any ABI or correctness-semantic prerequisite above remains unresolved.

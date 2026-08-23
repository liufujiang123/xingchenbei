# SparseFlashAttention design

Status: scalar correctness baseline builds on CANN 8.5, passes the local A3 matrix, and reaches CANNJudge Kernel result comparison; numerical/semantic correctness is now the active issue.

## 1. Contract boundary

The checked-in `workspace/code/` is the current contest template. Its public interface is authoritative for submission. The task statement defines the required SparseFlashAttention mathematics.

The implementation must preserve:

- operator registration `SparseFlashAttention`;
- kernel entrypoint `sparse_flash_attention`;
- all input/output/attribute names, order, optionality, defaults, dtype declarations, and required filenames.

The earlier `DsaSfa` investigation is unrelated to this contest instance and is not an implementation constraint.

## 2. Mathematical dataflow

For one logical output row `(b, q, h)`:

```text
sparse_indices[b,q,0,:]
        |
        v
valid KV positions -----> gather K / V / key_rope
        |                         ^
        |                         |
query + query_rope --------------+
        |
        v
score_k = (Q.K_k + QRope.KRope_k) * scale
        |
        +--> sparse_mode mask
        |
        v
stable softmax over sparse positions
        |
        +--> optional row max / exp sum
        |
        v
weighted sum of gathered V
        |
        v
attention_out[b,q,h,:]
```

The content dot width is fixed at `512`; the RoPE dot width is fixed at `64`; all query heads share the single KV head.

## 3. Independent and serial axes

Independent axes:

- batch;
- query position;
- query head.

Serial dependencies inside one row:

- sparse positions participate in one shared softmax state;
- the 512 output features share the same sparse softmax weights.

The baseline therefore assigns whole `(B,Q,H)` rows to AIV cores. A core owns every write to its row, so no inter-core reduction or synchronization is required.

## 4. Baseline Host Tiling

Host Tiling currently:

1. uses the same required/optional Tensor accessors as the official template;
2. derives only the Kernel-required `B`, `Q_S`, `KV_S`, `Q_N`, and `sparse_size` runtime facts;
3. relies on the generated ACLNN/official OpDef boundary for public-interface matching instead of returning `GRAPH_FAILED` for redundant business validation;
4. reads all five attributes with template defaults when a pointer is absent;
5. records optional actual-length presence and internal dtype flags;
6. selects the official FP16/FP32 TilingKey;
7. launches `min(B*Q_S*Q_N, available_AIV_cores)` cores without auxiliary outputs and one core when auxiliary outputs are enabled;
8. advertises zero workspace because the kernel does not access GM workspace.

The TilingData contains only internal runtime facts and does not alter the public interface.

## 5. Baseline row algorithm

The implementation uses online stable softmax so it does not need to materialize the sparse score vector.

State for one row:

```text
m = running maximum score
l = running sum of exp(score - m)
acc[512] = running unnormalized weighted-value numerator
```

For each valid sparse key with new score `s`:

```text
m_new = max(m, s)
alpha = exp(m - m_new)        # zero for the first valid key
beta  = exp(s - m_new)
l_new = l * alpha + beta
acc   = acc * alpha + beta * V_k
```

After all effective sparse keys:

```text
attention_out = acc / l
softmax_max_out = m
softmax_sum_out = l
```

This is mathematically equivalent to stable softmax and avoids a score-sized temporary.

## 6. Baseline physical implementation

The first implementation deliberately favors transparency over performance:

- direct scalar GM reads for query/key/value/RoPE/index tensors;
- float32 dot-product accumulation;
- float32 online-softmax state;
- one reusable 512-float UB accumulator per active AIV core;
- the accumulator remains float32 for the entire sparse loop and is cast to output dtype only once after normalization;
- scalar GM output writes are cache-cleaned before a row completes;
- auxiliary-output mode is single-core to prevent adjacent float32 rows from being written back from stale per-core copies of the same cache line;
- scalar exponential approximation using ln(2) range reduction and an eighth-order Taylor polynomial;
- no UB queueing, no L1/L0 staging, no Cube matmul.

The corrected kernel accumulator resides in per-core UB and Host Tiling advertises no auxiliary workspace:

```text
workspace_bytes = 0
```

This is expected to be slow. It exists to establish compiler/API and semantic correctness before optimization.

## 7. Shape and dtype inference

The current `InferShape` and `InferDataType` callbacks intentionally return `GRAPH_SUCCESS` without touching optional outputs, exactly as in the authoritative official template. Output metadata is left to the generated ACLNN/template path. The public OpDef and TilingKey expose only official float16/float32; the statement's BF16 wording remains UNRESOLVED against the package.

## 8. Variable lengths and masking

Effective lengths are read from optional GM tensors when present, otherwise physical `Q_S/KV_S` are used. Values are clamped to the physical dimensions for memory safety.

Padded query rows currently produce zero attention output.

For `sparse_mode=3`, the baseline interprets right-down causal as right-aligning the effective Q/KV sequences:

```text
key_pos <= query_pos + actual_kv_len - actual_query_len
```

For `sparse_mode=0`, only index validity and actual KV length apply.

Invalid/out-of-range sparse indices are skipped. If no effective sparse key remains, the baseline returns zero attention output and, when requested, `max=-FLT_MAX`, `sum=0`. These edge conventions require evaluator confirmation.

## 9. Sparse block handling

The template supplies `sparse_indices` with a physical Q row for every query token. The baseline consumes the row belonging to the current `q` directly for all supported `sparse_block_size` values. It assumes the upstream index tensor already reflects shared block selection decisions.

No extra `q / sparse_block_size` remapping is introduced without evaluator evidence.

## 10. Precision policy

Current policy:

- query/key/value and RoPE loads are converted to float32 for arithmetic;
- internal BF16 conversion helpers remain in the kernel source for compatibility, but BF16 is not publicly selectable by the current OpDef/TilingKey;
- content and RoPE dot products accumulate in float32;
- the supplied `scale_value` is converted through float16 once in the kernel before use, matching the task statement's explicit scale-precision requirement;
- online-softmax max/sum are float32;
- the full 512-wide weighted-value numerator remains float32 in the per-core UB accumulator;
- `attention_out` is cast to query dtype only after final normalization.

The main remaining numerical approximation is scalar `exp`. If evaluator precision is close but not sufficient, the first precision change should replace this approximation with the Ascend Vector `Exp` path without changing the row algorithm.

## 11. Known correctness risks

| Risk | Current baseline choice | Next evidence/fix |
|---|---|---|
| right-down causal alignment | standard effective-length right alignment | verify with evaluator/reference |
| empty sparse row | zero output, max=-FLT_MAX, sum=0 | verify evaluator convention |
| padded query row | zero output | verify evaluator convention |
| block-wise selection | consume provided row directly | verify block-size cases |
| scalar exp approximation | high-order range-reduced polynomial | replace with Ascend Vector Exp if precision fails |
| package FP32 vs statement BF16 mismatch | follow the authoritative package at the public ABI | BF16 exposure remains UNRESOLVED until platform evidence changes |
| retired Host over-validation | permissive Host now reaches result comparison on all public cases | do not reintroduce rejection unless the official contract requires it for memory safety |

## 12. Local correctness evidence

The CANN 8.5 local-A3 flow builds a temporary source mirror whose only target changes are `ascend910b` to `ascend910_93` in CMake and OpDef registration. The official source remains `ascend910b`.

Direct ACLNN invocation loads the generated `libcust_opapi.so` and compares NPU results with an independent NumPy reference. The deterministic official-domain matrix passed 10/10 cases covering basic FP16 dataflow, shared KV heads, invalid sparse suffixes, nonzero RoPE, optional actual lengths, right-down causal mode with unequal sequence lengths, auxiliary outputs enabled/disabled, and FP32. No NaN or Inf values were observed.

The local failures that led to the retained fixes were:

1. direct scalar writes to the per-core GM workspace caused AIV error `507035`; moving the same float32 accumulator lifetime into a per-core UB `TBuf` removed the exception;
2. adjacent auxiliary rows written by different cores shared cache lines and overwrote one another during cache writeback; explicit output cache cleaning plus single-core auxiliary mode removed the false sharing.

## 13. Build and validation plan

Completed locally:

1. CANN 8.5 `ascend910b` compilation;
2. temporary target-only A3 build and direct ACLNN invocation;
3. independent-reference validation of nonzero RoPE, sparse modes 0/3, optional lengths, multiple query heads, optional LSE outputs, and official FP16/FP32 paths;
4. CANNJudge doctor identity check for contest/public/internal IDs;
5. a permissive-Host submission (`6a8a841282cffa8f16ab684b`) crossing GetWorkspace on all three public cases and returning Wrong Answer with precision ratios `0.134765625`, `0.216796875`, and `0.08203125`.

Host/Tiling dispatch is now unblocked. The next work must diagnose Kernel semantics/precision from the public ratios before any performance optimization.

## 14. Optimization directions after correctness

Once the scalar baseline passes, likely high-impact dimensions are:

- move Q/Q-RoPE into UB and vectorize the existing FP32 UB accumulator updates;
- aggregate contiguous sparse-index runs before GM copy;
- gather K/K-RoPE/V in sparse tiles;
- compute multiple sparse scores with Cube/Matmul/MMAD instead of scalar dots;
- use Vector `Exp/ReduceMax/ReduceSum` or a tiled online-softmax pipeline;
- overlap gather, Cube score calculation, Vector softmax, and V accumulation;
- tune ownership across query rows, heads, and sparse tiles;
- specialize full/tail sparse tiles through internal TilingKey regimes.

Each optimization must preserve the current contest-visible interface and re-pass correctness before promotion.

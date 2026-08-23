# SparseFlashAttention design

Status: contest ABI aligned; scalar correctness baseline implemented; build/validation pending.

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

1. validates rank-4 logical tensors and rank-1 optional actual-length tensors;
2. derives `B`, `Q_S`, `KV_S`, `Q_N`, and `sparse_size` from runtime shapes;
3. validates fixed dimensions `D=512`, `Dr=64`, `KV_N=1`;
4. validates `query/key/value` dtype agreement and the template's float16/float32 domain;
5. reads all five attributes;
6. records presence of optional actual-length tensors;
7. records independent RoPE dtype flags;
8. launches `min(B*Q_S*Q_N, available_AIV_cores)` cores;
9. requests `usedCoreNum * 512 * sizeof(float)` workspace bytes.

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
- one reusable 512-float workspace accumulator per active AIV core;
- the accumulator remains float32 for the entire sparse loop and is cast to output dtype only once after normalization;
- scalar exponential approximation using ln(2) range reduction and an eighth-order Taylor polynomial;
- no UB queueing, no L1/L0 staging, no Cube matmul.

The workspace is bounded by core count rather than sequence length:

```text
workspace_bytes = usedCoreNum * 512 * sizeof(float)
```

This is expected to be slow. It exists to establish compiler/API and semantic correctness before optimization.

## 7. Shape and dtype inference

`InferShape` sets:

```text
attention_out      = query.shape
softmax_max_out    = (B, 1, Q_S, Q_N)
softmax_sum_out    = (B, 1, Q_S, Q_N)
```

when optional output descriptors are present.

`InferDataType` sets `attention_out` to query dtype and both auxiliary outputs to float32.

The template-visible dtype declarations remain unchanged: float16/float32 for primary tensors and float32 for auxiliary outputs.

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
- content and RoPE dot products accumulate in float32;
- the supplied `scale_value` is converted through float16 once in the kernel before use, matching the task statement's explicit scale-precision requirement;
- online-softmax max/sum are float32;
- the full 512-wide weighted-value numerator remains float32 in the per-core workspace;
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
| float32 template path | implemented for content tensors | confirm platform actually tests it |
| BF16 statement/template mismatch | do not alter public template | follow contest template/evaluator |

## 12. Build and validation plan

Next actions on the server:

1. build this exact branch with the contest-compatible CANN 8.5/910B path already established by the mature harness;
2. use `ascendc-operator-compile-debug` only to make the smallest API/compiler fixes;
3. do not change the public interface or baseline mathematics to silence compiler errors;
4. add a local CPU/Python reference test for small shapes if the generated ACLNN package can be invoked;
5. validate nonzero RoPE, sparse-mode 0/3, optional lengths, multiple query heads, and optional LSE outputs;
6. only after correctness evidence, submit/benchmark and start performance work.

No build, runtime, or correctness result is claimed in this document yet.

## 13. Optimization directions after correctness

Once the scalar baseline passes, likely high-impact dimensions are:

- move Q/Q-RoPE and the FP32 output accumulator into UB;
- aggregate contiguous sparse-index runs before GM copy;
- gather K/K-RoPE/V in sparse tiles;
- compute multiple sparse scores with Cube/Matmul/MMAD instead of scalar dots;
- use Vector `Exp/ReduceMax/ReduceSum` or a tiled online-softmax pipeline;
- overlap gather, Cube score calculation, Vector softmax, and V accumulation;
- tune ownership across query rows, heads, and sparse tiles;
- specialize full/tail sparse tiles through internal TilingKey regimes.

Each optimization must preserve the current contest-visible interface and re-pass correctness before promotion.

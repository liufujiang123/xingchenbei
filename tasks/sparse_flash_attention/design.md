# SparseFlashAttention design

Status: the FP32-vectorized baseline builds for CANN 8.5 `ascend910b` and passes the full local A3 launch matrix. CANNJudge submission `6a8b30fa82cffa8f16c9857e` confirms a large runtime improvement but remains Wrong Answer, so a platform semantic/output-contract mismatch is still active.

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
6. selects the FP16/FP32/BF16 compatibility TilingKey;
7. uses `GetCoreNumAiv()`, then falls back to `GetCoreNum()`, and uses one core only when both platform counts are zero;
8. launches `min(B*Q_S*Q_N, available_cores)` row tasks without auxiliary outputs;
9. when auxiliary outputs are enabled, assigns aligned groups of eight rows to cores so one core owns every 32-byte float cache line and false sharing cannot occur;
10. advertises zero workspace because the kernel does not access GM workspace.

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

## 6. Retained physical implementation

The retained correctness-first performance baseline uses:

- one contiguous `DataCopy` for each 512-element Q row and one for its 64-element Q-RoPE row, performed once per logical output row;
- contiguous `DataCopy` bursts for each selected 512-element K/V row and 64-element K-RoPE row;
- FP16/BF16-to-FP32 Vector `Cast` on load; FP32 data is copied directly;
- Vector `Mul` plus FP32 `ReduceSum` for the 512-wide content dot and 64-wide RoPE dot;
- FP32 online-softmax state and a reusable 512-float UB numerator;
- Vector `Muls`/`Axpy` for the online weighted-value update;
- Vector `Exp` for constant-time exponential evaluation with respect to score magnitude;
- one final FP32 normalization, optional cast, and UB-to-GM `DataCopy` for `attention_out`;
- aligned eight-row auxiliary buffers and UB-to-GM `DataCopyPad` writes;
- explicit MTE2/Vector/MTE3 event dependencies before buffer reuse and output writeback;
- no GM workspace, no L1/L0 staging, no Cube matmul, and no double buffering.

Per-core UB holds raw and FP32 Q/Q-RoPE tiles, reusable raw and FP32 K/V/K-RoPE tiles, the FP32 numerator, dot product/reduction scratch, output conversion storage, and two eight-float auxiliary buffers. Host Tiling advertises:

```text
workspace_bytes = 0
```

This is expected to be slow. It exists to establish compiler/API and semantic correctness before optimization.

## 7. Shape and dtype inference

The current `InferShape` and `InferDataType` callbacks intentionally return `GRAPH_SUCCESS` without touching optional outputs, exactly as in the authoritative official template. Output metadata is left to the generated ACLNN/template path. The public OpDef and TilingKey expose a FP16/FP32/BF16 compatibility union: FP32 preserves the fresh-package interface, while BF16 covers the live statement. Whether CANNJudge accepts the added signature is platform-UNRESOLVED, although local CANN 8.5 build, GetWorkspaceSize, and true-launch evidence all pass.

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
- the full 512-wide weighted-value numerator remains float32 in the per-core UB accumulator;
- `attention_out` is cast to query dtype only after final normalization.

The exponential is evaluated by the Ascend Vector `Exp` instruction. All retained local comparisons use an independent float64 NumPy reference before casting to the output contract.

## 11. Known correctness risks

| Risk | Current baseline choice | Next evidence/fix |
|---|---|---|
| right-down causal alignment | standard effective-length right alignment | verify with evaluator/reference |
| empty sparse row | zero output, max=-FLT_MAX, sum=0 | verify evaluator convention |
| padded query row | zero output | verify evaluator convention |
| block-wise selection | consume provided row directly | verify block-size cases |
| long online-softmax accumulation | FP32 recurrence and Vector Exp | stress-tested through 4096 selected tokens; retain FP32 |
| package FP32 vs statement BF16 mismatch | retain FP32 and add a complete BF16 selector/binary/kernel path | platform acceptance of the three-dtype union remains UNRESOLVED |
| retired Host over-validation | permissive Host now reaches result comparison on all public cases | do not reintroduce rejection unless the official contract requires it for memory safety |

## 12. Local correctness evidence

The CANN 8.5 local-A3 flow builds a temporary source mirror whose only target changes are `ascend910b` to `ascend910_93` in CMake and OpDef registration. The official source remains `ascend910b`.

Direct ACLNN invocation loads the generated `libcust_opapi.so` and compares NPU results with an independent NumPy float64 reference. The deterministic matrix passes 11/11 cases covering basic FP16 dataflow, shared KV heads, invalid sparse suffixes, nonzero RoPE, optional actual lengths, right-down causal mode with unequal sequence lengths, auxiliary outputs enabled/disabled, FP32, and a true-launch BF16 case. The GetWorkspaceSize-only matrix passes 41/41, including BF16 executor creation without `561002`. The independent block-wise matrix passes 2/2 cases.

The retained stress matrix passes 15/15 true-launch checks covering `sparseBlockSize=1/2/4/8/16/32/64/128`, `QN=128`, `B=2/4`, `Q_S=1/>1`, FP16/FP32, modes 0/3, actual lengths, auxiliary outputs, and selected-token counts 256/1024/4096. A replicated-row probe compares a one-row launch with a 64-row multi-core launch and obtains bitwise-identical attention/max/sum outputs.

The local failures that led to retained fixes were:

1. direct scalar writes to the per-core GM workspace caused AIV error `507035`; moving the same float32 accumulator lifetime into a per-core UB `TBuf` removed the exception;
2. adjacent auxiliary rows written by different cores shared cache lines and overwrote one another during cache writeback; explicit output cache cleaning plus single-core auxiliary mode removed the false sharing.
3. an initial low-level block-reduction chain did not reduce 512 products to one scalar correctly; the supported FP32 `ReduceSum` API with dedicated scratch fixed the dot product;
4. the first UB-to-GM output candidate lacked a Vector-to-MTE3 dependency; an explicit event before `DataCopy` fixed stale/NaN output reads.

## 13. Build and validation plan

Completed locally for the retained vectorized implementation:

1. CANN 8.5 `ascend910b` compilation;
2. temporary target-only A3 build and direct ACLNN invocation;
3. independent-reference validation of nonzero RoPE, sparse modes 0/3, optional lengths, multiple query heads through 128, optional LSE outputs, FP16/FP32/BF16 paths, all block sizes, batches through four, and sparse sizes through 4096;
4. CANNJudge doctor identity check for contest/public/internal IDs;
5. a permissive-Host submission (`6a8a841282cffa8f16ab684b`) crossing GetWorkspace on all three public cases and returning Wrong Answer with precision ratios `0.134765625`, `0.216796875`, and `0.08203125`.

Host/Tiling dispatch is unblocked. The retained candidate passed local correctness and performance gates. Its controlled platform submission executed without Runtime Error and reduced the three public runtimes from approximately `1134/1018/2811 ms` to `26.92/28.22/32.70 ms`, while the precision ratios were `0.134765625/0.216796875/0`. The unchanged first two ratios prove the previous Wrong Answer was not caused only by slow execution; the third ratio regression remains unresolved.

## 14. Later optimization directions

Possible follow-up work after platform correctness evidence:

- coalesce multiple consecutive sparse rows into larger K/K-RoPE/V transfers where UB capacity permits;
- use double-buffered K/V staging to overlap MTE2 and Vector work;
- specialize common block sizes and full/tail regimes through internal templates without changing the public ABI;
- investigate Cube/Matmul only if profiler evidence shows the Vector dot path dominates and the packing overhead amortizes;
- pipeline multiple rows per core to overlap gather, score reduction, exponential, and value accumulation;
- tune core ownership separately for small-row and long-sparse regimes.

Each optimization must preserve the current contest-visible interface and re-pass correctness before promotion.

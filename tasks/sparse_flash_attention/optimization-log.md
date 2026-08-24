# SparseFlashAttention optimization log

Only measured experiments belong here.

| Candidate | Hypothesis | Correctness | Score | Decision | Evidence |
|---|---|---:|---:|---|---|
| scalar FP16/FP32 baseline | establish a contract-complete reference | local 10/10; platform Host RE | - | reject for platform | submission `6a8a5b8c82cffa8f16a62ace`; all 3 cases `GetWorkspaceSize=561002` |
| replace FP32 with BF16 | test whether statement dtype fixes Host rejection | local 10/10 including ACL_BF16; platform Host RE | - | reject hypothesis | submission `6a8a5d4382cffa8f16a6600e`; same pre-Kernel error |
| FP16/BF16/FP32 compatibility union | reconcile fresh package and live statement | local 11/11; platform Host RE | - | retain locally; block further submit | submission `6a8a5eb282cffa8f16a685e1`; same pre-Kernel error |
| official interface + strict custom Host | test whether restoring OpDef/Infer/TilingKey is sufficient | local official-domain 10/10; platform Host RE | - | reject Host policy | submission `6a8a817782cffa8f16ab04b9`; all 3 cases still `561002` |
| official interface + permissive Host | remove non-template Host rejection while retaining Kernel-required metadata | local 10/10; GetWorkspace matrix 39/39; platform reaches comparison | - | retain | submission `6a8a841282cffa8f16ab684b`; Wrong Answer, public precision `0.134765625/0.216796875/0.08203125` |
| 910B UB/Vector refactor | replace scalar GM access/dot/exp/value accumulation with contiguous copies and FP32 Vector operations; restore safe aux multicore | local base 10/10, block 2/2, stress 15/15, single/multi-core bitwise match; platform WA | local QN128 `0.094620 ms`; sparse 256/1024/4096 `0.269960/1.028820/4.035800 ms`; platform `26.92/28.22/32.70 ms` | retain performance work; platform semantics unresolved | submission `6a8b30fa82cffa8f16c9857e`; no Runtime Error; public precision `0.134765625/0.216796875/0`; platform speedup versus `6a8aa89b82cffa8f16b24329` is `42.1x/36.1x/86.0x` |

## 2026-08-23 retained 910B vector candidate

- Hypothesis: scalar GM Q/K/V/RoPE/output access and scalar 512-wide arithmetic dominate the correctness baseline; whole-row UB transfers and FP32 Vector primitives should reduce instruction and GM transaction overhead without changing the online-softmax recurrence.
- Host: `GetCoreNumAiv()==0` now falls back to the generic core count before one core. Auxiliary output uses aligned groups of eight rows, allowing multiple cores while retaining exclusive 32-byte cache-line ownership.
- Kernel: Q/Q-RoPE are loaded once per row; selected K/V/K-RoPE use contiguous GM-to-UB copies; dot products use `Mul+ReduceSum`; the numerator uses `Muls/Axpy`; exponential uses Vector `Exp`; the final output uses UB-to-GM `DataCopy`.
- Precision: every dot, maximum, exponential sum, and 512-wide numerator stays FP32. The output is cast once at the end. Local float64-reference coverage passes through 4096 selected tokens.
- Rejected implementation details inside this direction: the first low-level block-reduction sequence and the first output copy dependency were incorrect. They were replaced by supported `ReduceSum` scratch and an explicit Vector-to-MTE3 event; the overall architecture was retained.
- Measurement method: CANN 8.5 A3 target-only mirror, identical device/cases, ACL runtime start/end events around the launched operator, one warm-up excluded, seven measured iterations, median reported. Python allocation and tensor-construction time is excluded from `device_ms`.
- Platform result: submission `6a8b30fa82cffa8f16c9857e` ran all three public cases without Runtime Error. The first two precision ratios are unchanged from the scalar submission and the third fell from `0.08203125` to `0`; therefore performance was a real bottleneck but not the root cause of Wrong Answer.

## Submission experience

- Bind submissions to the verified triple `(contest_id, public_problem_id, internal_problem_id)`. A successful login or problem-name match alone is insufficient.
- Count a submission only after the platform returns a submission ID. The three IDs above are valid submissions; earlier HTTP failures without an ID would not be.
- Download and hash the current package immediately before the first upload. Filename problem numbers are not provenance.
- Treat `GetWorkspaceSize` failure as a Host/interface/tiling failure. It is not evidence about Kernel math, precision, or performance because the Kernel never launched.
- Change one contract hypothesis at a time. The unchanged `561002` across the FP32-only, BF16-only, and union variants disproves dtype selection as the sole cause.
- Restoring only OpDef/Infer/TilingKey was insufficient. Removing redundant custom Host validation crossed GetWorkspace on all public cases, proving the failure class was Host over-validation; the platform does not reveal which retired predicate was the individual trigger.
- Stop uploading when the supported result lacks enough information to distinguish remaining Host checks. Repeated blind submissions only consume quota.
- Keep target-clean build and local-device validation separate: official source stays `ascend910b`; the A3 test flow changes only target declarations in a temporary mirror.

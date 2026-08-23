# SparseFlashAttention optimization log

Only measured experiments belong here.

| Candidate | Hypothesis | Correctness | Score | Decision | Evidence |
|---|---|---:|---:|---|---|
| scalar FP16/FP32 baseline | establish a contract-complete reference | local 10/10; platform Host RE | - | reject for platform | submission `6a8a5b8c82cffa8f16a62ace`; all 3 cases `GetWorkspaceSize=561002` |
| replace FP32 with BF16 | test whether statement dtype fixes Host rejection | local 10/10 including ACL_BF16; platform Host RE | - | reject hypothesis | submission `6a8a5d4382cffa8f16a6600e`; same pre-Kernel error |
| FP16/BF16/FP32 compatibility union | reconcile fresh package and live statement | local 11/11; platform Host RE | - | retain locally; block further submit | submission `6a8a5eb282cffa8f16a685e1`; same pre-Kernel error |

## Submission experience

- Bind submissions to the verified triple `(contest_id, public_problem_id, internal_problem_id)`. A successful login or problem-name match alone is insufficient.
- Count a submission only after the platform returns a submission ID. The three IDs above are valid submissions; earlier HTTP failures without an ID would not be.
- Download and hash the current package immediately before the first upload. Filename problem numbers are not provenance.
- Treat `GetWorkspaceSize` failure as a Host/interface/tiling failure. It is not evidence about Kernel math, precision, or performance because the Kernel never launched.
- Change one contract hypothesis at a time. The unchanged `561002` across the FP32-only, BF16-only, and union variants disproves dtype selection as the sole cause.
- Stop uploading when the supported result lacks enough information to distinguish remaining Host checks. Repeated blind submissions only consume quota.
- Keep target-clean build and local-device validation separate: official source stays `ascend910b`; the A3 test flow changes only target declarations in a temporary mirror.

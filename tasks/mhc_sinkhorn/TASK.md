# MhcSinkhorn task contract

Status: **EXECUTABLE ABI RESOLVED — independent correctness baseline validated**.

This document records both the documentation conflict discovered on 2026-08-22
UTC and its later resolution from user-supplied code that has passed the current
platform. The conflict history remains visible; implementation follows the
single proven executable ABI below.

## Objective and current scope

The objective is to implement and optimize the competition-provided
`MhcSinkhorn` Ascend C operator while preserving the exact platform-visible
interface and mathematical semantics. The current correctness baseline was
implemented independently from the fresh scaffold and frozen ABI; the poorer
historical passing kernel is neither required nor used as an implementation
baseline.

## Platform execution ABI

The user-confirmed platform execution facts are the highest-priority ABI
evidence for this task. The immutable public interface is:

| Position | Name | Kind | Required/optional | Dtype / format | Shape/value contract |
|---:|---|---|---|---|---|
| 0 | `logits` | input tensor | required | FP16 or FP32 / ND | rank `>=2`, shape `[...,N,N]`, `N in {4,6,8}` |
| 1 | `mask` | input tensor | optional | same dtype as `logits` / ND | absent, scalar, or full tensor with element count equal to `logits` |
| 0 | `weights` | output tensor | required | same dtype and shape as `logits` / ND | one Sinkhorn result per independent matrix |
| 0 | `iterations` | attr | optional, default `20` | int | one initial column normalization plus `iterations-1` row/column pairs; retained range `[1,100]` |
| 1 | `eps` | attr | optional, default `1e-6` | float | placement follows the numerical sequence below; no additional legal range is asserted |

Operator registration remains `MhcSinkhorn`. The kernel GM argument topology
remains:

```text
logits, mask, weights, workspace, tiling
```

No public input/output may be added, removed, reordered, renamed, or have its
optionality changed. In particular, there is no executable
`x -> output, normOut?, sumOut?` path and `normOut/sumOut` must not be added.

### Shape and task domain

For `logits.shape = [...,N,N]`:

```text
matrixCount = product(all leading dimensions)
```

Every `N x N` matrix is an independent logical task. Leading dimensions may be
flattened internally without changing the public shape. The Sinkhorn steps
within one matrix retain their serial dependency. The supported domain is every
rank `>=2` shape satisfying the square trailing dimensions and
`N in {4,6,8}`; public testcases do not narrow it further.

### Mask modes

Host Tiling must derive exactly one internal mask regime:

```text
MASK_MODE=0: optional mask is absent
MASK_MODE=1: mask contains one scalar element
MASK_MODE=2: mask element count equals logits element count
```

The mask dtype must equal the logits dtype. Other present mask element counts
are outside the supported contract. `MASK_MODE` is internal tiling/template
metadata, not a public attribute.

### Numerical baseline

The retained non-conflicting operation order is the correctness reference:

```text
logits
  -> convert to FP32
  -> add optional mask
  -> stable row softmax
  -> add eps to each softmax result
  -> divide by (FP32 column sum + eps)
  -> remaining serial row/column Sinkhorn iterations
  -> cast to logits dtype
  -> weights
```

FP16 and FP32 storage are both required; sensitive calculation remains FP32.
The stable softmax denominator itself does not receive eps. Each remaining row
or column normalization divides by `(FP32 sum + eps)`. Changing mask
placement, eps placement, reduction order, iteration semantics,
or the FP32 compute path is a numerical reformulation requiring an isolated
correctness experiment, not an ordinary performance refactor.

### Compile-time regimes

The legal AIV-only specialization dimensions are:

```text
DT_LOGITS in {FP16, FP32}
N         in {4, 6, 8}
MASK_MODE in {0, 1, 2}
```

Internal specialization may change after validation, but it must continue to
implement this single public ABI.

## Evidence and authority

### Source-control safety check

The required start-of-phase check reported branch `task/mhc-expand`. Existing
changes under `tasks/mhc_expand/` were treated as unrelated user work and left
untouched. MhcSinkhorn work remains scoped to `tasks/mhc_sinkhorn/` and its
task-local Harness configuration.

### Verified CANNJudge identity

The formal submission target was verified against all three identity fields:

| Field | Current value |
|---|---|
| contest ID | `6a7bf087a52e0f540a88e167` |
| verified public problem ID | `302` |
| internal problem ID | `6a7c2267a52e0f540a8a02bd` |
| name / canonical name | `mhcsinkhorn` / `mhcsinkhorn` |
| title | `【B组中等题】mHC-Sinkhorn 算子` |
| version | `v1` (`version_no=1`) |
| CANN version | `8.5.0` |
| template type | `custom_template` |
| score mode | `0`, which the current CANNJudge UI labels `按部分测试点` |
| baseline timing | disabled (`use_baseline=false`) |
| ranking representative | `latest` |

The metadata response does not expose a `soc_version` field. The current
official package sets `ASCEND_COMPUTE_UNIT ascend910b`, so `ascend910b` is the
package-supported target evidence; it must not be upgraded into a stronger
claim about an absent metadata field.

The checked-in statement
`B组中等题_mHC-Sinkhorn迭代算子题目.md` matches the verified problem 302
statement hash recorded by the repository. Therefore:

- **problem statement matches platform: yes**;
- this confirms the statement contents, but it does not resolve the statement
  versus package ABI conflict below.

### Verified official package

The package for internal problem `6a7c2267a52e0f540a8a02bd` was inspected
without overwriting the workspace. It confirms `logits, mask? -> weights`,
FP16/FP32, attributes `iterations=20` and `eps=1e-6`, and target
`ascend910b`.

The wrong same-name contest remains relevant only as incident history. Its
contest ID was `6a75f67feff6447fafb8cbfd`, public problem ID `284`, and
internal problem ID `6a7961aea52e0f540a7bfdea`. Public reads succeeded, but
submission returned HTTP 403 because the account had no permission in that
contest. Those identifiers must never be used as current MhcSinkhorn metadata.

## Documentation conflict: platform statement

This is a faithful historical extraction of the current platform statement.
It is documentation evidence only where it does not conflict with the proven
platform execution ABI.

### Public objects described by the statement

| Position | Name | Kind | Required/optional | Dtype / format | Shape or value domain |
|---:|---|---|---|---|---|
| 0 | `x` | input tensor | required by the described computation | FP32 / ND | rank 3 `(T,n,n)` or rank 4 `(B,S,n,n)`; `n in {4,6,8}` |
| 0 | `output` | output tensor | required | FP32 / ND | identical logical shape to `x` |
| 1 | `normOut` | output tensor | optional; statement says a null pointer means absent | FP32 / aligned device storage | logical `(2*numIters,T,n,n)` or `(2*numIters,B,S,n,n)` |
| 2 | `sumOut` | output tensor | optional; statement says a null pointer means absent | FP32 / aligned device storage | logical `(2*numIters,T,n)` or `(2*numIters,B,S,n)` |
| - | `eps` | float32 attr | optionality/default are not formally declared; statement recommends `1e-6` | FP32 scalar | legal range is not stated |
| - | `numIters` | int64 attr | optionality/default are not formally declared; statement recommends `20` | integer scalar | `[1,100]`; outside the range is invalid |

The statement lists `eps` before `numIters`, but it does not provide an OpDef
or generated ACLNN signature. Attribute ABI ordering and defaults therefore
cannot be inferred solely from the prose.

### Logical and physical output storage

Let `matrix_count=T` for rank 3 and `matrix_count=B*S` for rank 4, and let

```text
n_align = ceil(n / 8) * 8 = 8 for n in {4,6,8}.
```

The statement distinguishes tensor logical shapes from physical storage sizes:

```text
output:  logical/compact size = matrix_count * n * n
normOut: logical size = 2*numIters * matrix_count * n * n
         physical size = 2*numIters * matrix_count * n * n_align
sumOut:  logical size = 2*numIters * matrix_count * n
         physical size = 2*numIters * matrix_count * n_align
```

Thus `n=4` requires twice the logical element count for each optional output.
The prose says the `n` dimension is aligned, but does not provide a generated
API/storage-shape descriptor that completely fixes allocation and stride
representation. That missing ABI evidence is material and must be resolved
before Host InferShape or output allocation is implemented.

### Mathematical contract described by the statement

For every independent `n x n` matrix, with all operations in FP32:

```text
normOut[0] = softmax(x, dim=-1) + eps
sumOut[1]  = sum(normOut[0], dim=-2, keepdim=true) + eps
normOut[1] = normOut[0] / sumOut[1]

for i = 1 .. numIters-1:
    sumOut[2*i]    = sum(normOut[2*i-1], dim=-1, keepdim=true) + eps
    normOut[2*i]   = normOut[2*i-1] / sumOut[2*i]
    sumOut[2*i+1]  = sum(normOut[2*i], dim=-2, keepdim=true) + eps
    normOut[2*i+1] = normOut[2*i] / sumOut[2*i+1]

output = normOut[2*numIters-1]
```

`sumOut[0]` is allocated but undefined/unwritten and is explicitly excluded
from intermediate correctness checking. For `numIters=1`, the output is
`normOut[1]` after the initial softmax and column normalization.

The operation order is contractual: `softmax(x) + eps` is not
`softmax(x + eps)`, and each denominator is `sum(current_norm) + eps`.
The statement also requires deterministic behavior and says inputs containing
`inf`, `-inf`, or `nan` produce `nan` at corresponding positions; exact
propagation beyond that wording remains reference-sensitive.

## Documentation conflict: official package ABI

The official package exposes a materially different OpDef and kernel
entrypoint from the prose statement:

| Position | Package name | Kind | Required/optional | Dtype / format | Shape/storage evidence |
|---:|---|---|---|---|---|
| 0 | `logits` | input tensor | required | FP16 or FP32 / ND | executable domain is `[...,N,N]` |
| 1 | `mask` | input tensor | optional | logits dtype / ND | absent, scalar, or full tensor |
| 0 | `weights` | output tensor | required | logits dtype / ND | same shape as `logits` |
| 0 | `iterations` | attr | optional, default `20` | int | semantic range is not validated |
| 1 | `eps` | attr | optional, default `1e-6` | float | semantic range is not validated |

Additional package-visible facts:

- registration/operator class: `MhcSinkhorn`;
- target configuration: `ascend910b`;
- TilingKey includes FP16 and FP32 `DT_LOGITS` regimes;
- TilingData: exactly one `uint32_t length` field;
- Host Tiling reads required input 0, optional input 1, attrs in
  `iterations, eps` order, sets all available AIV cores, and requests zero
  workspace;
- kernel physical parameters are
  `logits, mask, weights, workspace, tiling`;
- `InferShape` and `InferDataType` return success without setting outputs;
- there are no `output`, `normOut`, or `sumOut` parameters in the package.

No public evidence authorizes any of these guesses:

```text
logits == x
mask == normOut or sumOut
weights == output
iterations == numIters
```

The first three in particular cannot represent the statement's 1-input,
3-output ABI by renaming alone.

## Official package versus retained implementation

The retained implementation preserves the verified package-facing symbols and
ABI. Host inference, validation, tiling data, dtype/N/mask specializations, and
the optimized kernel are internal implementation work; none creates a second
statement-ABI path.

## Documentation conflict and resolution

The retained **PLATFORM/TASK DOCUMENTATION CONFLICT** is:

- current platform statement: `x -> output, optional normOut, optional sumOut`,
  FP32, rank 3/4 Sinkhorn semantics;
- official package/executable ABI: `logits, optional mask -> weights`, FP16 or
  FP32, with no Sinkhorn intermediate outputs.

This no longer blocks implementation. It is resolved operationally as follows:

1. platform-passing code determines the executable ABI and callable symbols;
2. the statement may inform non-conflicting Sinkhorn mathematics only;
3. statement-only `x/output/normOut/sumOut`, padded optional-output storage, and
   related shape rules do not become executable requirements;
4. only one `logits, mask? -> weights` implementation is maintained;
5. no variable-name mapping is invented to make the two documents appear
   consistent.

The historical conflict is retained to prevent future interface drift. Hidden
testcases must not be accessed or inferred.

## Performance objective

The platform uses partial-testpoint scoring (`score_mode=0`), has baseline
timing disabled, and exposes no per-test baseline. Correctness remains the gate
for every candidate. Performance claims distinguish formal CANNJudge evidence
from local `Ascend910_9382` proxy measurements.

## Interface rule

The executable ABI defined above is immutable. Host Tiling, blockDim, logical
task ownership, matrix batching, TilingData, internal TilingKey, UB layout,
TQue/TBuf ownership, buffer lifetimes, Vector instruction organization,
reduction implementation, pipeline, workspace, and scalar/control overhead may
change only while the public ABI, required functional domain, and validated
numerical contract remain intact.

# MhcSinkhorn local A3 performance — harness_20260822t163809z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 12.9806 | 12.980601 | 0.0770 | 12.0078, 12.8376, 12.9156, 12.9648, 12.9806, 12.9954, 13.1432, 13.3336, 13.4558 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.8832 | 3.220800 | 0.3105 | 12.6138, 12.7410, 12.8300, 12.8380, 12.8832, 12.9292, 12.9826, 13.2680, 13.7070 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.2720 | 0.276500 | 3.6166 | 13.0476, 13.2236, 13.2252, 13.2260, 13.2720, 13.2872, 13.3076, 13.5212, 13.5688 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.5970 | 0.070818 | 14.1208 | 13.3880, 13.4050, 13.4220, 13.5890, 13.5970, 13.7790, 13.8310, 14.0830, 14.3700 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 34.9260 | 0.045477 | 21.9893 | 34.9060, 34.9150, 34.9220, 34.9250, 34.9260, 34.9330, 34.9340, 34.9500, 35.9440 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.1340 | 0.068406 | 14.6185 | 12.5424, 13.0604, 13.0676, 13.1232, 13.1340, 13.1388, 13.1684, 13.1960, 13.6816 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.0200 | 0.067813 | 14.7465 | 12.6988, 12.8332, 12.8612, 12.8964, 13.0200, 13.1372, 13.1964, 13.2364, 13.4552 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 43.1100 | 0.224531 | 4.4537 | 41.7880, 41.8080, 43.0480, 43.0720, 43.1100, 43.1620, 43.1840, 43.1960, 43.9060 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.3564 | 0.100815 | 9.9192 | 19.2836, 19.3264, 19.3288, 19.3324, 19.3564, 19.5952, 19.6040, 19.6232, 19.6236 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 21.4560 | 0.111750 | 8.9485 | 21.4420, 21.4470, 21.4470, 21.4520, 21.4560, 21.4600, 21.4680, 21.4860, 21.9120 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.3510 | 0.069536 | 14.3809 | 13.0160, 13.0370, 13.1070, 13.1320, 13.3510, 13.4580, 13.6950, 13.8460, 13.9330 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.3750 | 0.069661 | 14.3551 | 13.1890, 13.2590, 13.3000, 13.3370, 13.3750, 13.5070, 13.7900, 14.0280, 14.1330 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.4370 | 0.069984 | 14.2889 | 13.3400, 13.4180, 13.4240, 13.4350, 13.4370, 13.8640, 14.0960, 14.2060, 14.5680 |

## Matrix-count amortization

- matrix_count=1: 12.9806 us (12.980601 us/matrix)
- matrix_count=4: 12.8832 us (3.220800 us/matrix)
- matrix_count=48: 13.2720 us (0.276500 us/matrix)
- matrix_count=192: 13.5970 us (0.070818 us/matrix)
- matrix_count=768: 34.9260 us (0.045477 us/matrix)

## Iteration cost model

- iterations=1: 13.1340 us (0.068406 us/matrix)
- iterations=5: 13.0200 us (0.067813 us/matrix)
- iterations=20: 13.5970 us (0.070818 us/matrix)
- iterations=100: 43.1100 us (0.224531 us/matrix)

Least-squares proxy: `T(iterations) = 10.7093 us + iterations * 0.3176 us`, R²=0.974025.

## Orthogonal comparisons

### N

- n=4: 19.3564 us (0.100815 us/matrix)
- n=6: 21.4560 us (0.111750 us/matrix)
- n=8: 13.5970 us (0.070818 us/matrix)

### DType

- dtype=float16: 13.3510 us (0.069536 us/matrix)
- dtype=float32: 13.5970 us (0.070818 us/matrix)

### Mask

- mask_mode=absent: 13.5970 us (0.070818 us/matrix)
- mask_mode=full: 13.4370 us (0.069984 us/matrix)
- mask_mode=scalar: 13.3750 us (0.069661 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

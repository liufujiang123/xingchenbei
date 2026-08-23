# MhcSinkhorn retained baseline — harness_20260822t154740z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2554 | 13.255399 | 0.0754 | 13.0778, 13.0892, 13.1468, 13.2016, 13.2554, 13.3950, 13.4062, 13.4168, 13.6702 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0802 | 3.270050 | 0.3058 | 12.7992, 12.9462, 12.9572, 13.0082, 13.0802, 13.1200, 13.1580, 13.2326, 13.7254 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.2556 | 0.276158 | 3.6211 | 13.0712, 13.1060, 13.2180, 13.2436, 13.2556, 13.3328, 13.3632, 13.3660, 13.3724 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 22.7640 | 0.118563 | 8.4344 | 22.7530, 22.7540, 22.7550, 22.7560, 22.7640, 22.7670, 22.7680, 22.7750, 23.7570 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 76.3780 | 0.099451 | 10.0553 | 76.3500, 76.3550, 76.3660, 76.3680, 76.3780, 76.3820, 76.4000, 76.4150, 76.8460 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.6244 | 0.070960 | 14.0924 | 12.7624, 13.1620, 13.3688, 13.3880, 13.6244, 13.6532, 13.9764, 14.0172, 14.0296 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.1940 | 0.068719 | 14.5521 | 12.9208, 13.1412, 13.1448, 13.1604, 13.1940, 13.2356, 13.3500, 13.3928, 13.8304 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 92.4240 | 0.481375 | 2.0774 | 92.3980, 92.4100, 92.4120, 92.4180, 92.4240, 92.4380, 92.4380, 92.4720, 92.4800 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 68.8908 | 0.358806 | 2.7870 | 68.8816, 68.8864, 68.8876, 68.8896, 68.8908, 68.9116, 68.9184, 69.0364, 69.2576 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 158.3560 | 0.824771 | 1.2125 | 157.7170, 157.7220, 158.2740, 158.3420, 158.3560, 158.3630, 158.3700, 158.3740, 158.3790 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 23.4550 | 0.122161 | 8.1859 | 23.4040, 23.4270, 23.4330, 23.4550, 23.4550, 23.4560, 23.4690, 23.4750, 25.6140 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 23.4660 | 0.122219 | 8.1821 | 23.4520, 23.4640, 23.4640, 23.4650, 23.4660, 23.4750, 23.4870, 23.5180, 23.6820 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 23.3090 | 0.121401 | 8.2372 | 23.2900, 23.2940, 23.2990, 23.3060, 23.3090, 23.3200, 23.3300, 23.3360, 23.3400 |

## Matrix-count amortization

- matrix_count=1: 13.2554 us (13.255399 us/matrix)
- matrix_count=4: 13.0802 us (3.270050 us/matrix)
- matrix_count=48: 13.2556 us (0.276158 us/matrix)
- matrix_count=192: 22.7640 us (0.118563 us/matrix)
- matrix_count=768: 76.3780 us (0.099451 us/matrix)

## Iteration cost model

- iterations=1: 13.6244 us (0.070960 us/matrix)
- iterations=5: 13.1940 us (0.068719 us/matrix)
- iterations=20: 22.7640 us (0.118563 us/matrix)
- iterations=100: 92.4240 us (0.481375 us/matrix)

Least-squares proxy: `T(iterations) = 9.6261 us + iterations * 0.8214 us`, R²=0.995058.

## Orthogonal comparisons

### N

- n=4: 68.8908 us (0.358806 us/matrix)
- n=6: 158.3560 us (0.824771 us/matrix)
- n=8: 22.7640 us (0.118563 us/matrix)

### DType

- dtype=float16: 23.4550 us (0.122161 us/matrix)
- dtype=float32: 22.7640 us (0.118563 us/matrix)

### Mask

- mask_mode=absent: 22.7640 us (0.118563 us/matrix)
- mask_mode=full: 23.3090 us (0.121401 us/matrix)
- mask_mode=scalar: 23.4660 us (0.122219 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

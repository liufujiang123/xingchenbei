# MhcSinkhorn local A3 performance — harness_20260822t164946z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.4480 | 13.448000 | 0.0744 | 13.2350, 13.2416, 13.2480, 13.3516, 13.4480, 13.4694, 13.6204, 13.7620, 13.8966 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.1892 | 3.297300 | 0.3033 | 13.0438, 13.0748, 13.1544, 13.1740, 13.1892, 13.1992, 13.2090, 13.2708, 13.3670 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.4052 | 0.279275 | 3.5807 | 12.8508, 13.0420, 13.2716, 13.3908, 13.4052, 13.4536, 13.5520, 13.5876, 13.6080 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.4260 | 0.069927 | 14.3006 | 12.9560, 13.1170, 13.1320, 13.3460, 13.4260, 13.7140, 13.8660, 14.7210, 14.8910 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 19.3380 | 0.025180 | 39.7146 | 19.3100, 19.3120, 19.3220, 19.3240, 19.3380, 19.3390, 19.3510, 19.3580, 19.3780 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4808 | 0.070213 | 14.2425 | 12.8048, 13.3936, 13.4340, 13.4704, 13.4808, 13.5168, 13.5632, 13.6604, 13.7376 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4540 | 0.070073 | 14.2708 | 13.0336, 13.1500, 13.3876, 13.4068, 13.4540, 13.4580, 13.5556, 13.5848, 13.7288 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 28.4300 | 0.148073 | 6.7534 | 27.1900, 28.3620, 28.3720, 28.4120, 28.4300, 28.4520, 28.4720, 28.5180, 29.2500 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.9092 | 0.103694 | 9.6438 | 19.8860, 19.8968, 19.9028, 19.9076, 19.9092, 19.9160, 19.9372, 19.9552, 19.9764 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 22.0550 | 0.114870 | 8.7055 | 22.0180, 22.0380, 22.0390, 22.0520, 22.0550, 22.0610, 22.0750, 22.0900, 22.1100 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.5080 | 0.070354 | 14.2138 | 13.3090, 13.4230, 13.4580, 13.5040, 13.5080, 14.1750, 14.2550, 14.2710, 14.8550 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.8770 | 0.072276 | 13.8358 | 13.1810, 13.2080, 13.2330, 13.2880, 13.8770, 13.9280, 14.1060, 14.2460, 14.2940 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 14.0300 | 0.073073 | 13.6850 | 13.1810, 13.2120, 13.2870, 13.4770, 14.0300, 14.1510, 14.4780, 14.5030, 14.5370 |

## Matrix-count amortization

- matrix_count=1: 13.4480 us (13.448000 us/matrix)
- matrix_count=4: 13.1892 us (3.297300 us/matrix)
- matrix_count=48: 13.4052 us (0.279275 us/matrix)
- matrix_count=192: 13.4260 us (0.069927 us/matrix)
- matrix_count=768: 19.3380 us (0.025180 us/matrix)

## Iteration cost model

- iterations=1: 13.4808 us (0.070213 us/matrix)
- iterations=5: 13.4540 us (0.070073 us/matrix)
- iterations=20: 13.4260 us (0.069927 us/matrix)
- iterations=100: 28.4300 us (0.148073 us/matrix)

Least-squares proxy: `T(iterations) = 12.1956 us + iterations * 0.1588 us`, R²=0.967927.

## Orthogonal comparisons

### N

- n=4: 19.9092 us (0.103694 us/matrix)
- n=6: 22.0550 us (0.114870 us/matrix)
- n=8: 13.4260 us (0.069927 us/matrix)

### DType

- dtype=float16: 13.5080 us (0.070354 us/matrix)
- dtype=float32: 13.4260 us (0.069927 us/matrix)

### Mask

- mask_mode=absent: 13.4260 us (0.069927 us/matrix)
- mask_mode=full: 14.0300 us (0.073073 us/matrix)
- mask_mode=scalar: 13.8770 us (0.072276 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

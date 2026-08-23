# MhcSinkhorn local A3 performance — candidate-core-batch8

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2452 | 13.245200 | 0.0755 | 12.8424, 12.9882, 13.2280, 13.2430, 13.2452, 13.3198, 13.3960, 13.6786, 13.8184 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0736 | 3.268400 | 0.3060 | 12.8770, 12.8992, 12.9548, 12.9798, 13.0736, 13.1016, 13.1930, 13.1932, 13.3032 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5024 | 0.281300 | 3.5549 | 13.2744, 13.2952, 13.4020, 13.4272, 13.5024, 13.5968, 13.6092, 13.6140, 13.6252 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.3660 | 0.069615 | 14.3648 | 12.9210, 13.3000, 13.3140, 13.3410, 13.3660, 13.4430, 13.5320, 13.9490, 14.7430 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 16.9170 | 0.022027 | 45.3981 | 16.8870, 16.9040, 16.9140, 16.9160, 16.9170, 16.9300, 16.9500, 16.9620, 17.0040 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4196 | 0.069894 | 14.3074 | 13.1168, 13.3716, 13.3864, 13.4188, 13.4196, 13.4664, 13.5020, 13.5548, 13.5604 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.3768 | 0.069671 | 14.3532 | 12.9792, 13.1524, 13.2724, 13.3056, 13.3768, 13.3800, 13.4108, 13.4132, 13.4460 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 38.9100 | 0.202656 | 4.9345 | 38.6660, 38.8040, 38.8360, 38.8880, 38.9100, 39.0660, 39.0800, 39.1280, 40.1380 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.5196 | 0.086040 | 11.6226 | 16.4856, 16.4892, 16.5008, 16.5164, 16.5196, 16.5276, 16.5508, 16.5828, 16.6012 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.9020 | 0.098448 | 10.1577 | 18.6490, 18.8770, 18.8930, 18.9010, 18.9020, 18.9050, 18.9070, 18.9460, 18.9770 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.4360 | 0.069979 | 14.2900 | 13.0770, 13.3610, 13.4160, 13.4230, 13.4360, 13.8790, 14.0120, 14.2370, 14.2590 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.3810 | 0.069693 | 14.3487 | 13.1660, 13.1790, 13.3090, 13.3360, 13.3810, 13.5230, 13.6850, 13.7240, 13.9650 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.5710 | 0.070682 | 14.1478 | 13.3810, 13.4080, 13.4990, 13.5380, 13.5710, 13.6250, 14.0920, 14.2250, 14.7670 |

## Matrix-count amortization

- matrix_count=1: 13.2452 us (13.245200 us/matrix)
- matrix_count=4: 13.0736 us (3.268400 us/matrix)
- matrix_count=48: 13.5024 us (0.281300 us/matrix)
- matrix_count=192: 13.3660 us (0.069615 us/matrix)
- matrix_count=768: 16.9170 us (0.022027 us/matrix)

## Iteration cost model

- iterations=1: 13.4196 us (0.069894 us/matrix)
- iterations=5: 13.3768 us (0.069671 us/matrix)
- iterations=20: 13.3660 us (0.069615 us/matrix)
- iterations=100: 38.9100 us (0.202656 us/matrix)

Least-squares proxy: `T(iterations) = 11.2414 us + iterations * 0.2707 us`, R²=0.968420.

## Orthogonal comparisons

### N

- n=4: 16.5196 us (0.086040 us/matrix)
- n=6: 18.9020 us (0.098448 us/matrix)
- n=8: 13.3660 us (0.069615 us/matrix)

### DType

- dtype=float16: 13.4360 us (0.069979 us/matrix)
- dtype=float32: 13.3660 us (0.069615 us/matrix)

### Mask

- mask_mode=absent: 13.3660 us (0.069615 us/matrix)
- mask_mode=full: 13.5710 us (0.070682 us/matrix)
- mask_mode=scalar: 13.3810 us (0.069693 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

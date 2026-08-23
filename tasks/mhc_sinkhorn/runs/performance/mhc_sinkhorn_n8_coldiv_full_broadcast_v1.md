# MhcSinkhorn local A3 performance — n8_coldiv_full_broadcast_v1

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.5068 | 13.506800 | 0.0740 | 13.2686, 13.2960, 13.4144, 13.4400, 13.5068, 13.5296, 13.6386, 13.9494, 14.2610 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.2796 | 3.319900 | 0.3012 | 13.1138, 13.1458, 13.2558, 13.2634, 13.2796, 13.3204, 13.3612, 13.3652, 13.4454 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.6036 | 0.283408 | 3.5285 | 13.3232, 13.5288, 13.5316, 13.5472, 13.6036, 13.6712, 13.6936, 13.7468, 14.0400 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 14.0480 | 0.073167 | 13.6674 | 13.5760, 13.6660, 13.8680, 14.0120, 14.0480, 14.4190, 14.5300, 14.5960, 14.9460 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 14.6340 | 0.019055 | 52.4805 | 14.4670, 14.4790, 14.6190, 14.6330, 14.6340, 14.7690, 14.7840, 14.9400, 15.0350 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.6576 | 0.071133 | 14.0581 | 12.8788, 13.2964, 13.3908, 13.4060, 13.6576, 13.7784, 13.7868, 13.8116, 14.1788 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4792 | 0.070204 | 14.2442 | 13.4116, 13.4216, 13.4364, 13.4516, 13.4792, 13.5060, 13.5120, 13.5364, 13.5852 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 33.7120 | 0.175583 | 5.6953 | 33.6280, 33.6700, 33.6800, 33.7060, 33.7120, 33.7440, 33.7740, 33.7800, 34.4180 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.8384 | 0.087700 | 11.4025 | 16.7304, 16.8176, 16.8280, 16.8380, 16.8384, 16.8484, 16.8496, 16.8652, 16.8928 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.0040 | 0.098979 | 10.1031 | 18.8140, 18.8210, 18.8650, 18.9850, 19.0040, 19.0160, 19.0410, 19.0550, 19.1160 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 14.4810 | 0.075422 | 13.2588 | 13.4900, 13.5490, 13.6430, 13.6810, 14.4810, 14.5140, 14.6000, 14.6010, 14.6630 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.5580 | 0.070615 | 14.1614 | 13.4600, 13.5430, 13.5440, 13.5480, 13.5580, 13.5980, 13.9400, 14.3380, 14.5440 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.7570 | 0.071651 | 13.9565 | 13.5520, 13.5640, 13.6130, 13.7090, 13.7570, 14.5080, 14.5370, 14.5870, 14.5950 |

## Matrix-count amortization

- matrix_count=1: 13.5068 us (13.506800 us/matrix)
- matrix_count=4: 13.2796 us (3.319900 us/matrix)
- matrix_count=48: 13.6036 us (0.283408 us/matrix)
- matrix_count=192: 14.0480 us (0.073167 us/matrix)
- matrix_count=768: 14.6340 us (0.019055 us/matrix)

## Iteration cost model

- iterations=1: 13.6576 us (0.071133 us/matrix)
- iterations=5: 13.4792 us (0.070204 us/matrix)
- iterations=20: 14.0480 us (0.073167 us/matrix)
- iterations=100: 33.7120 us (0.175583 us/matrix)

Least-squares proxy: `T(iterations) = 12.0214 us + iterations * 0.2128 us`, R²=0.975574.

## Orthogonal comparisons

### N

- n=4: 16.8384 us (0.087700 us/matrix)
- n=6: 19.0040 us (0.098979 us/matrix)
- n=8: 14.0480 us (0.073167 us/matrix)

### DType

- dtype=float16: 14.4810 us (0.075422 us/matrix)
- dtype=float32: 14.0480 us (0.073167 us/matrix)

### Mask

- mask_mode=absent: 14.0480 us (0.073167 us/matrix)
- mask_mode=full: 13.7570 us (0.071651 us/matrix)
- mask_mode=scalar: 13.5580 us (0.070615 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

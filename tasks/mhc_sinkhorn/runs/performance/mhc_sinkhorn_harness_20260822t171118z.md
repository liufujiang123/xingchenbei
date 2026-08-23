# MhcSinkhorn local A3 performance — harness_20260822t171118z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2806 | 13.280600 | 0.0753 | 13.0222, 13.0528, 13.1560, 13.2626, 13.2806, 13.2934, 13.3362, 13.6062, 13.9212 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.3008 | 3.325200 | 0.3007 | 12.8082, 13.0752, 13.1116, 13.1806, 13.3008, 13.4182, 13.4862, 13.4968, 13.5612 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.3912 | 0.278983 | 3.5844 | 13.0124, 13.0884, 13.1056, 13.1948, 13.3912, 13.4604, 13.5480, 13.6072, 13.7100 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.4360 | 0.069979 | 14.2900 | 13.0810, 13.2140, 13.2500, 13.2950, 13.4360, 13.5820, 13.8580, 14.0390, 14.1160 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 16.5270 | 0.021520 | 46.4694 | 16.5020, 16.5060, 16.5230, 16.5260, 16.5270, 16.5300, 16.5380, 16.5390, 16.6440 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4100 | 0.069844 | 14.3177 | 12.7796, 13.2608, 13.3104, 13.3160, 13.4100, 13.4216, 13.4552, 13.5128, 13.7500 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.2040 | 0.068771 | 14.5410 | 12.7372, 13.0728, 13.1496, 13.1852, 13.2040, 13.2172, 13.2396, 13.2952, 13.3500 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.0460 | 0.140865 | 7.0990 | 27.0100, 27.0180, 27.0400, 27.0460, 27.0460, 27.0480, 27.0500, 27.0720, 27.1020 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.2880 | 0.084833 | 11.7878 | 16.2276, 16.2760, 16.2816, 16.2876, 16.2880, 16.2936, 16.3060, 16.3068, 16.5436 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.5040 | 0.096375 | 10.3761 | 18.4540, 18.4700, 18.4770, 18.4840, 18.5040, 18.5250, 18.5540, 18.6780, 18.7150 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.2450 | 0.068984 | 14.4960 | 12.9910, 13.1190, 13.1570, 13.2070, 13.2450, 13.4830, 13.7890, 13.9620, 14.1480 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.2400 | 0.068958 | 14.5015 | 12.7790, 12.9050, 12.9130, 13.1740, 13.2400, 13.4070, 13.5310, 13.9970, 14.9060 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.6140 | 0.070906 | 14.1031 | 13.1400, 13.1490, 13.1900, 13.2870, 13.6140, 13.6600, 13.9880, 14.0880, 14.3590 |

## Matrix-count amortization

- matrix_count=1: 13.2806 us (13.280600 us/matrix)
- matrix_count=4: 13.3008 us (3.325200 us/matrix)
- matrix_count=48: 13.3912 us (0.278983 us/matrix)
- matrix_count=192: 13.4360 us (0.069979 us/matrix)
- matrix_count=768: 16.5270 us (0.021520 us/matrix)

## Iteration cost model

- iterations=1: 13.4100 us (0.069844 us/matrix)
- iterations=5: 13.2040 us (0.068771 us/matrix)
- iterations=20: 13.4360 us (0.069979 us/matrix)
- iterations=100: 27.0460 us (0.140865 us/matrix)

Least-squares proxy: `T(iterations) = 12.1921 us + iterations * 0.1455 us`, R²=0.970870.

## Orthogonal comparisons

### N

- n=4: 16.2880 us (0.084833 us/matrix)
- n=6: 18.5040 us (0.096375 us/matrix)
- n=8: 13.4360 us (0.069979 us/matrix)

### DType

- dtype=float16: 13.2450 us (0.068984 us/matrix)
- dtype=float32: 13.4360 us (0.069979 us/matrix)

### Mask

- mask_mode=absent: 13.4360 us (0.069979 us/matrix)
- mask_mode=full: 13.6140 us (0.070906 us/matrix)
- mask_mode=scalar: 13.2400 us (0.068958 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

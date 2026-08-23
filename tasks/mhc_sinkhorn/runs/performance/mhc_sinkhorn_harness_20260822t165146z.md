# MhcSinkhorn local A3 performance — harness_20260822t165146z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2920 | 13.292000 | 0.0752 | 13.0194, 13.1388, 13.1596, 13.2632, 13.2920, 13.4860, 13.7380, 13.7934, 13.8938 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0848 | 3.271200 | 0.3057 | 12.8566, 12.8720, 12.8866, 12.9612, 13.0848, 13.0954, 13.2794, 13.2916, 13.5128 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.1840 | 0.274667 | 3.6408 | 12.9808, 12.9860, 13.0108, 13.0536, 13.1840, 13.1972, 13.3868, 13.5332, 13.7056 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.5070 | 0.070349 | 14.2149 | 12.9420, 13.1270, 13.3140, 13.4440, 13.5070, 13.5670, 13.8020, 13.8600, 14.3710 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 23.6270 | 0.030764 | 32.5052 | 23.0020, 23.0160, 23.5590, 23.6260, 23.6270, 23.6430, 23.6540, 23.6700, 23.6900 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.3436 | 0.069498 | 14.3889 | 13.0960, 13.1544, 13.2004, 13.2464, 13.3436, 13.4024, 13.4612, 13.4692, 13.8668 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.0920 | 0.068188 | 14.6654 | 13.0288, 13.0320, 13.0340, 13.0592, 13.0920, 13.1756, 13.2360, 13.3048, 13.3968 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.1420 | 0.141365 | 7.0739 | 27.1000, 27.1260, 27.1320, 27.1420, 27.1420, 27.1500, 27.1500, 27.1640, 28.3520 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.7616 | 0.087300 | 11.4548 | 16.6928, 16.7064, 16.7320, 16.7564, 16.7616, 16.7908, 16.9880, 17.0432, 17.1248 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.4590 | 0.101349 | 9.8669 | 19.4310, 19.4330, 19.4430, 19.4560, 19.4590, 19.4610, 19.4830, 19.5290, 19.5910 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.5090 | 0.070359 | 14.2127 | 13.3550, 13.4260, 13.4540, 13.4950, 13.5090, 13.5600, 13.5880, 13.6680, 13.8470 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.6940 | 0.071323 | 14.0207 | 13.3170, 13.3790, 13.4500, 13.6580, 13.6940, 13.8660, 14.0860, 14.1600, 14.6330 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.7980 | 0.071865 | 13.9151 | 13.5620, 13.5850, 13.6710, 13.7110, 13.7980, 13.9160, 14.0610, 14.0830, 14.5850 |

## Matrix-count amortization

- matrix_count=1: 13.2920 us (13.292000 us/matrix)
- matrix_count=4: 13.0848 us (3.271200 us/matrix)
- matrix_count=48: 13.1840 us (0.274667 us/matrix)
- matrix_count=192: 13.5070 us (0.070349 us/matrix)
- matrix_count=768: 23.6270 us (0.030764 us/matrix)

## Iteration cost model

- iterations=1: 13.3436 us (0.069498 us/matrix)
- iterations=5: 13.0920 us (0.068188 us/matrix)
- iterations=20: 13.5070 us (0.070349 us/matrix)
- iterations=100: 27.1420 us (0.141365 us/matrix)

Least-squares proxy: `T(iterations) = 12.1367 us + iterations * 0.1471 us`, R²=0.974013.

## Orthogonal comparisons

### N

- n=4: 16.7616 us (0.087300 us/matrix)
- n=6: 19.4590 us (0.101349 us/matrix)
- n=8: 13.5070 us (0.070349 us/matrix)

### DType

- dtype=float16: 13.5090 us (0.070359 us/matrix)
- dtype=float32: 13.5070 us (0.070349 us/matrix)

### Mask

- mask_mode=absent: 13.5070 us (0.070349 us/matrix)
- mask_mode=full: 13.7980 us (0.071865 us/matrix)
- mask_mode=scalar: 13.6940 us (0.071323 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

# MhcSinkhorn local A3 performance — n8_iter_unroll2_v1

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2240 | 13.224000 | 0.0756 | 12.8222, 13.0090, 13.0470, 13.1198, 13.2240, 13.2876, 13.3712, 13.6260, 13.6856 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.9644 | 3.241100 | 0.3085 | 12.7066, 12.7726, 12.8342, 12.8664, 12.9644, 13.0042, 13.0094, 13.0332, 13.1506 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5524 | 0.282342 | 3.5418 | 13.2712, 13.2852, 13.4184, 13.4328, 13.5524, 13.5892, 13.6404, 13.7056, 17.0396 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.7700 | 0.071719 | 13.9434 | 13.4130, 13.4360, 13.5340, 13.6810, 13.7700, 13.8900, 14.0400, 14.5390, 14.7030 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 17.0700 | 0.022227 | 44.9912 | 16.9640, 17.0410, 17.0510, 17.0610, 17.0700, 17.0720, 17.0900, 17.3880, 17.4260 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4216 | 0.069904 | 14.3053 | 13.1524, 13.2384, 13.2624, 13.4072, 13.4216, 13.4724, 13.7800, 13.9412, 13.9852 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4108 | 0.069848 | 14.3168 | 13.0644, 13.2672, 13.2688, 13.3792, 13.4108, 13.4456, 13.5332, 13.6344, 13.6556 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 28.1540 | 0.146635 | 6.8196 | 27.9780, 28.0900, 28.1140, 28.1320, 28.1540, 28.1640, 28.1680, 28.1800, 28.9180 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.5316 | 0.086102 | 11.6141 | 16.5072, 16.5216, 16.5236, 16.5276, 16.5316, 16.5392, 16.5432, 16.5540, 16.5748 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.0240 | 0.099083 | 10.0925 | 18.9450, 18.9710, 19.0130, 19.0200, 19.0240, 19.0620, 19.0700, 19.1330, 19.1550 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.6190 | 0.070932 | 14.0980 | 13.2540, 13.5200, 13.5220, 13.5510, 13.6190, 14.1920, 14.2870, 14.6140, 14.7090 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.6150 | 0.070911 | 14.1021 | 13.1600, 13.3400, 13.4400, 13.5500, 13.6150, 13.7160, 14.0810, 14.4590, 14.5030 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.6360 | 0.071021 | 14.0804 | 13.1860, 13.3250, 13.5110, 13.5710, 13.6360, 13.8320, 13.8710, 14.2190, 14.2810 |

## Matrix-count amortization

- matrix_count=1: 13.2240 us (13.224000 us/matrix)
- matrix_count=4: 12.9644 us (3.241100 us/matrix)
- matrix_count=48: 13.5524 us (0.282342 us/matrix)
- matrix_count=192: 13.7700 us (0.071719 us/matrix)
- matrix_count=768: 17.0700 us (0.022227 us/matrix)

## Iteration cost model

- iterations=1: 13.4216 us (0.069904 us/matrix)
- iterations=5: 13.4108 us (0.069848 us/matrix)
- iterations=20: 13.7700 us (0.071719 us/matrix)
- iterations=100: 28.1540 us (0.146635 us/matrix)

Least-squares proxy: `T(iterations) = 12.2841 us + iterations * 0.1557 us`, R²=0.976147.

## Orthogonal comparisons

### N

- n=4: 16.5316 us (0.086102 us/matrix)
- n=6: 19.0240 us (0.099083 us/matrix)
- n=8: 13.7700 us (0.071719 us/matrix)

### DType

- dtype=float16: 13.6190 us (0.070932 us/matrix)
- dtype=float32: 13.7700 us (0.071719 us/matrix)

### Mask

- mask_mode=absent: 13.7700 us (0.071719 us/matrix)
- mask_mode=full: 13.6360 us (0.071021 us/matrix)
- mask_mode=scalar: 13.6150 us (0.070911 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

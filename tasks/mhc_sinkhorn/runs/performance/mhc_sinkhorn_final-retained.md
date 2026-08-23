# MhcSinkhorn local A3 performance — final-retained

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2068 | 13.206800 | 0.0757 | 12.9364, 13.0794, 13.1696, 13.1814, 13.2068, 13.2532, 13.2712, 13.4856, 14.0888 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.7796 | 3.194900 | 0.3130 | 12.6668, 12.6918, 12.7172, 12.7418, 12.7796, 12.7964, 12.8896, 12.9584, 13.1154 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 12.7908 | 0.266475 | 3.7527 | 12.4348, 12.4892, 12.6888, 12.7136, 12.7908, 13.1308, 13.2092, 13.4280, 13.5876 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 12.9400 | 0.067396 | 14.8377 | 12.7100, 12.7500, 12.8680, 12.9110, 12.9400, 13.0240, 13.1080, 13.1190, 13.6610 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 17.1280 | 0.022302 | 44.8389 | 16.9740, 16.9790, 17.0830, 17.1180, 17.1280, 17.1310, 17.1430, 17.2020, 17.4560 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 12.6020 | 0.065635 | 15.2357 | 12.3480, 12.5556, 12.5684, 12.5972, 12.6020, 12.6328, 12.6664, 12.8292, 12.9608 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 12.5880 | 0.065563 | 15.2526 | 12.4916, 12.5148, 12.5396, 12.5428, 12.5880, 12.6044, 12.6456, 12.6720, 12.8816 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.5440 | 0.143458 | 6.9707 | 27.4940, 27.5060, 27.5200, 27.5340, 27.5440, 27.5500, 28.1360, 28.4400, 28.4420 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 17.1372 | 0.089256 | 11.2037 | 16.9920, 17.0788, 17.0904, 17.1264, 17.1372, 17.1440, 17.2004, 17.2216, 17.2252 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.8830 | 0.098349 | 10.1679 | 18.7480, 18.7890, 18.8310, 18.8580, 18.8830, 18.9360, 18.9360, 19.2180, 19.2590 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.1860 | 0.068677 | 14.5609 | 13.0140, 13.0930, 13.1440, 13.1530, 13.1860, 13.2120, 13.2620, 13.5760, 13.6600 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.1290 | 0.068380 | 14.6241 | 12.6660, 12.7170, 12.7270, 13.0330, 13.1290, 13.2070, 13.2940, 13.3280, 13.4550 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.3250 | 0.069401 | 14.4090 | 12.7750, 13.0650, 13.2590, 13.2770, 13.3250, 13.3410, 13.3720, 13.5480, 13.6840 |

## Matrix-count amortization

- matrix_count=1: 13.2068 us (13.206800 us/matrix)
- matrix_count=4: 12.7796 us (3.194900 us/matrix)
- matrix_count=48: 12.7908 us (0.266475 us/matrix)
- matrix_count=192: 12.9400 us (0.067396 us/matrix)
- matrix_count=768: 17.1280 us (0.022302 us/matrix)

## Iteration cost model

- iterations=1: 12.6020 us (0.065635 us/matrix)
- iterations=5: 12.5880 us (0.065563 us/matrix)
- iterations=20: 12.9400 us (0.067396 us/matrix)
- iterations=100: 27.5440 us (0.143458 us/matrix)

Least-squares proxy: `T(iterations) = 11.4425 us + iterations * 0.1580 us`, R²=0.975870.

## Orthogonal comparisons

### N

- n=4: 17.1372 us (0.089256 us/matrix)
- n=6: 18.8830 us (0.098349 us/matrix)
- n=8: 12.9400 us (0.067396 us/matrix)

### DType

- dtype=float16: 13.1860 us (0.068677 us/matrix)
- dtype=float32: 12.9400 us (0.067396 us/matrix)

### Mask

- mask_mode=absent: 12.9400 us (0.067396 us/matrix)
- mask_mode=full: 13.3250 us (0.069401 us/matrix)
- mask_mode=scalar: 13.1290 us (0.068380 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

# MhcSinkhorn retained baseline — retained_baseline

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 67.5726 | 67.572598 | 0.0148 | 67.5532, 67.5698, 67.5698, 67.5708, 67.5726, 67.5782, 67.6282, 67.6426, 68.6414 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 67.7210 | 16.930250 | 0.0591 | 67.7118, 67.7182, 67.7188, 67.7198, 67.7210, 67.7248, 67.7522, 67.8622, 69.8924 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 71.2964 | 1.485342 | 0.6732 | 71.1016, 71.1024, 71.2768, 71.2936, 71.2964, 71.3192, 71.3268, 71.3348, 71.4320 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 270.5790 | 1.409266 | 0.7096 | 270.5690, 270.5700, 270.5730, 270.5780, 270.5790, 270.5860, 270.5880, 270.5890, 271.0840 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 1068.0640 | 1.390708 | 0.7191 | 1068.0180, 1068.0450, 1068.0510, 1068.0610, 1068.0640, 1068.0820, 1068.4960, 1068.5570, 1068.5900 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 25.2032 | 0.131267 | 7.6181 | 25.1568, 25.1748, 25.1764, 25.1940, 25.2032, 25.2088, 25.2112, 25.2124, 25.2220 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 76.9184 | 0.400617 | 2.4962 | 76.6924, 76.7012, 76.8732, 76.9084, 76.9184, 76.9192, 76.9208, 76.9328, 76.9956 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 1297.4900 | 6.757761 | 0.1480 | 1296.5500, 1296.5840, 1296.7000, 1296.7220, 1297.4900, 1297.6900, 1297.7040, 1297.7280, 1297.7500 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 69.0972 | 0.359881 | 2.7787 | 68.9000, 69.0536, 69.0900, 69.0916, 69.0972, 69.0984, 69.1224, 69.1264, 69.1272 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 158.1630 | 0.823766 | 1.2139 | 158.0740, 158.1450, 158.1550, 158.1600, 158.1630, 158.1640, 158.1730, 158.1740, 158.4310 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 271.1420 | 1.412198 | 0.7081 | 270.6120, 270.6530, 270.6610, 271.1410, 271.1420, 271.1500, 271.1600, 271.1660, 271.1750 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 271.5640 | 1.414396 | 0.7070 | 271.0330, 271.4630, 271.4750, 271.4870, 271.5640, 271.5670, 271.5710, 271.5890, 271.5970 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 271.4700 | 1.413906 | 0.7073 | 271.0510, 271.4420, 271.4530, 271.4700, 271.4700, 271.4810, 271.5040, 271.5190, 271.5940 |

## Matrix-count amortization

- matrix_count=1: 67.5726 us (67.572598 us/matrix)
- matrix_count=4: 67.7210 us (16.930250 us/matrix)
- matrix_count=48: 71.2964 us (1.485342 us/matrix)
- matrix_count=192: 270.5790 us (1.409266 us/matrix)
- matrix_count=768: 1068.0640 us (1.390708 us/matrix)

## Iteration cost model

- iterations=1: 25.2032 us (0.131267 us/matrix)
- iterations=5: 76.9184 us (0.400617 us/matrix)
- iterations=20: 270.5790 us (1.409266 us/matrix)
- iterations=100: 1297.4900 us (6.757761 us/matrix)

Least-squares proxy: `T(iterations) = 12.8365 us + iterations * 12.8480 us`, R²=0.999999.

## Orthogonal comparisons

### N

- n=4: 69.0972 us (0.359881 us/matrix)
- n=6: 158.1630 us (0.823766 us/matrix)
- n=8: 270.5790 us (1.409266 us/matrix)

### DType

- dtype=float16: 271.1420 us (1.412198 us/matrix)
- dtype=float32: 270.5790 us (1.409266 us/matrix)

### Mask

- mask_mode=absent: 270.5790 us (1.409266 us/matrix)
- mask_mode=full: 271.4700 us (1.413906 us/matrix)
- mask_mode=scalar: 271.5640 us (1.414396 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

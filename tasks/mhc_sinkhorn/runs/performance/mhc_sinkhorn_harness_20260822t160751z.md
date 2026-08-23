# MhcSinkhorn local A3 performance — harness_20260822t160751z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2420 | 13.242000 | 0.0755 | 12.9566, 12.9624, 13.0050, 13.1338, 13.2420, 13.2582, 13.3348, 13.6466, 16.9776 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.9120 | 3.228000 | 0.3098 | 12.6920, 12.7482, 12.8090, 12.8500, 12.9120, 12.9744, 12.9934, 13.0300, 13.4106 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.1496 | 0.273950 | 3.6503 | 12.8660, 12.9776, 13.0132, 13.0420, 13.1496, 13.1984, 13.2664, 13.2684, 13.4964 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 23.3130 | 0.121422 | 8.2357 | 23.2660, 23.2800, 23.3030, 23.3110, 23.3130, 23.3180, 23.3210, 23.3230, 23.3270 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 76.2990 | 0.099348 | 10.0657 | 76.2800, 76.2930, 76.2970, 76.2980, 76.2990, 76.3100, 76.3470, 76.9490, 76.9780 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.6824 | 0.071262 | 14.0326 | 12.9488, 13.4376, 13.5652, 13.6076, 13.6824, 13.9476, 13.9656, 13.9916, 14.5912 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.3980 | 0.069781 | 14.3305 | 12.9228, 13.1196, 13.1456, 13.3756, 13.3980, 13.4040, 13.4844, 13.6412, 13.7964 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 93.6220 | 0.487615 | 2.0508 | 92.4180, 93.2560, 93.2940, 93.2960, 93.6220, 93.6220, 93.6360, 93.6380, 93.7780 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 22.4964 | 0.117169 | 8.5347 | 22.4720, 22.4816, 22.4888, 22.4912, 22.4964, 22.4992, 22.5032, 22.5156, 22.5260 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 27.2140 | 0.141740 | 7.0552 | 27.1990, 27.2100, 27.2120, 27.2130, 27.2140, 27.2440, 27.2670, 27.8610, 27.8890 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 23.0620 | 0.120115 | 8.3254 | 22.7100, 22.7390, 22.7810, 23.0620, 23.0620, 23.1420, 23.1440, 23.1620, 23.2340 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 24.0510 | 0.125266 | 7.9830 | 24.0010, 24.0400, 24.0410, 24.0490, 24.0510, 24.0550, 24.0580, 24.0630, 24.0660 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 23.8350 | 0.124141 | 8.0554 | 23.2200, 23.2850, 23.7850, 23.8310, 23.8350, 23.8390, 23.8390, 23.8530, 23.8640 |

## Matrix-count amortization

- matrix_count=1: 13.2420 us (13.242000 us/matrix)
- matrix_count=4: 12.9120 us (3.228000 us/matrix)
- matrix_count=48: 13.1496 us (0.273950 us/matrix)
- matrix_count=192: 23.3130 us (0.121422 us/matrix)
- matrix_count=768: 76.2990 us (0.099348 us/matrix)

## Iteration cost model

- iterations=1: 13.6824 us (0.071262 us/matrix)
- iterations=5: 13.3980 us (0.069781 us/matrix)
- iterations=20: 23.3130 us (0.121422 us/matrix)
- iterations=100: 93.6220 us (0.487615 us/matrix)

Least-squares proxy: `T(iterations) = 9.7938 us + iterations * 0.8321 us`, R²=0.995594.

## Orthogonal comparisons

### N

- n=4: 22.4964 us (0.117169 us/matrix)
- n=6: 27.2140 us (0.141740 us/matrix)
- n=8: 23.3130 us (0.121422 us/matrix)

### DType

- dtype=float16: 23.0620 us (0.120115 us/matrix)
- dtype=float32: 23.3130 us (0.121422 us/matrix)

### Mask

- mask_mode=absent: 23.3130 us (0.121422 us/matrix)
- mask_mode=full: 23.8350 us (0.124141 us/matrix)
- mask_mode=scalar: 24.0510 us (0.125266 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

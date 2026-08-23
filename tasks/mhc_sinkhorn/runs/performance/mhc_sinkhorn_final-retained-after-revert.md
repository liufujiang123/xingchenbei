# MhcSinkhorn local A3 performance — final-retained-after-revert

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 12.8558 | 12.855800 | 0.0778 | 12.7786, 12.8050, 12.8098, 12.8162, 12.8558, 12.9834, 13.0048, 13.0366, 13.3098 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.8150 | 3.203750 | 0.3121 | 12.7466, 12.7520, 12.7998, 12.8000, 12.8150, 12.8628, 12.8848, 12.9184, 13.0370 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 12.9976 | 0.270783 | 3.6930 | 12.7368, 12.8360, 12.9296, 12.9612, 12.9976, 13.1540, 13.2252, 13.2752, 13.3312 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.0410 | 0.067922 | 14.7228 | 12.9090, 12.9510, 12.9830, 12.9920, 13.0410, 13.0980, 13.1710, 13.2870, 13.4810 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 16.8820 | 0.021982 | 45.4922 | 16.8570, 16.8630, 16.8680, 16.8720, 16.8820, 16.8830, 16.8880, 16.9020, 16.9580 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 12.9640 | 0.067521 | 14.8102 | 12.7756, 12.8192, 12.9020, 12.9548, 12.9640, 13.0156, 13.0868, 13.1124, 13.2900 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 12.8228 | 0.066785 | 14.9733 | 12.7588, 12.7768, 12.7824, 12.7940, 12.8228, 12.8324, 12.9416, 12.9580, 12.9612 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.8380 | 0.144990 | 6.8970 | 27.0280, 27.0340, 27.0360, 27.0500, 27.8380, 27.8580, 27.8960, 27.9180, 28.7280 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.6028 | 0.086473 | 11.5643 | 16.5144, 16.5412, 16.5584, 16.5752, 16.6028, 16.6288, 16.6416, 16.7088, 20.8468 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.1430 | 0.099703 | 10.0298 | 18.9790, 19.1310, 19.1400, 19.1410, 19.1430, 19.1450, 19.1570, 19.1910, 19.2240 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.2440 | 0.068979 | 14.4971 | 12.9470, 13.0010, 13.2310, 13.2350, 13.2440, 13.6580, 15.7520, 16.6770, 17.4990 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.1180 | 0.068323 | 14.6364 | 12.7980, 12.8600, 12.9330, 12.9620, 13.1180, 13.1210, 13.1600, 13.3400, 13.7760 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.0430 | 0.067932 | 14.7205 | 12.6560, 12.8520, 12.9500, 12.9940, 13.0430, 13.1170, 13.1230, 13.2660, 13.4340 |

## Matrix-count amortization

- matrix_count=1: 12.8558 us (12.855800 us/matrix)
- matrix_count=4: 12.8150 us (3.203750 us/matrix)
- matrix_count=48: 12.9976 us (0.270783 us/matrix)
- matrix_count=192: 13.0410 us (0.067922 us/matrix)
- matrix_count=768: 16.8820 us (0.021982 us/matrix)

## Iteration cost model

- iterations=1: 12.9640 us (0.067521 us/matrix)
- iterations=5: 12.8228 us (0.066785 us/matrix)
- iterations=20: 13.0410 us (0.067922 us/matrix)
- iterations=100: 27.8380 us (0.144990 us/matrix)

Least-squares proxy: `T(iterations) = 11.6820 us + iterations * 0.1582 us`, R²=0.971422.

## Orthogonal comparisons

### N

- n=4: 16.6028 us (0.086473 us/matrix)
- n=6: 19.1430 us (0.099703 us/matrix)
- n=8: 13.0410 us (0.067922 us/matrix)

### DType

- dtype=float16: 13.2440 us (0.068979 us/matrix)
- dtype=float32: 13.0410 us (0.067922 us/matrix)

### Mask

- mask_mode=absent: 13.0410 us (0.067922 us/matrix)
- mask_mode=full: 13.0430 us (0.067932 us/matrix)
- mask_mode=scalar: 13.1180 us (0.068323 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

# MhcSinkhorn local A3 performance — harness_20260822t164200z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.0806 | 13.080601 | 0.0764 | 13.0008, 13.0158, 13.0346, 13.0630, 13.0806, 13.1242, 13.4800, 13.8250, 14.6060 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.8010 | 3.200250 | 0.3125 | 11.5538, 11.9410, 12.5554, 12.6648, 12.8010, 12.8246, 12.8550, 12.9454, 13.1870 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 12.7724 | 0.266092 | 3.7581 | 12.6592, 12.6596, 12.7148, 12.7548, 12.7724, 12.8756, 13.0688, 13.1276, 13.3876 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.4070 | 0.069828 | 14.3209 | 13.0480, 13.3590, 13.3680, 13.3900, 13.4070, 13.4240, 13.4910, 13.7180, 14.0590 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 22.4090 | 0.029178 | 34.2719 | 22.3980, 22.4000, 22.4050, 22.4090, 22.4090, 22.4170, 22.4240, 23.0180, 23.0230 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.0804 | 0.068127 | 14.6784 | 12.9512, 12.9824, 13.0160, 13.0440, 13.0804, 13.1364, 13.1848, 13.2760, 13.3192 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 12.9056 | 0.067217 | 14.8773 | 12.6760, 12.7388, 12.7868, 12.8800, 12.9056, 12.9104, 13.0296, 13.1292, 13.2788 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 26.6960 | 0.139042 | 7.1921 | 26.6600, 26.6740, 26.6840, 26.6900, 26.6960, 26.7080, 26.7540, 28.0740, 28.1020 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.0916 | 0.099435 | 10.0568 | 19.0356, 19.0428, 19.0532, 19.0564, 19.0916, 19.1852, 19.3584, 19.3668, 19.3832 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 21.1740 | 0.110281 | 9.0677 | 20.9640, 20.9650, 21.0070, 21.0560, 21.1740, 21.3910, 21.5810, 21.6630, 21.6810 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.1660 | 0.068573 | 14.5830 | 12.9260, 13.0880, 13.1250, 13.1270, 13.1660, 13.2030, 13.4850, 13.4900, 13.5150 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.1780 | 0.068635 | 14.5697 | 12.9620, 13.0020, 13.0470, 13.1400, 13.1780, 13.2130, 13.3210, 13.3940, 13.6490 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.1150 | 0.068307 | 14.6397 | 12.9410, 12.9700, 13.0420, 13.0900, 13.1150, 13.2160, 13.2250, 13.2830, 13.4820 |

## Matrix-count amortization

- matrix_count=1: 13.0806 us (13.080601 us/matrix)
- matrix_count=4: 12.8010 us (3.200250 us/matrix)
- matrix_count=48: 12.7724 us (0.266092 us/matrix)
- matrix_count=192: 13.4070 us (0.069828 us/matrix)
- matrix_count=768: 22.4090 us (0.029178 us/matrix)

## Iteration cost model

- iterations=1: 13.0804 us (0.068127 us/matrix)
- iterations=5: 12.9056 us (0.067217 us/matrix)
- iterations=20: 13.4070 us (0.069828 us/matrix)
- iterations=100: 26.6960 us (0.139042 us/matrix)

Least-squares proxy: `T(iterations) = 11.9680 us + iterations * 0.1446 us`, R²=0.977083.

## Orthogonal comparisons

### N

- n=4: 19.0916 us (0.099435 us/matrix)
- n=6: 21.1740 us (0.110281 us/matrix)
- n=8: 13.4070 us (0.069828 us/matrix)

### DType

- dtype=float16: 13.1660 us (0.068573 us/matrix)
- dtype=float32: 13.4070 us (0.069828 us/matrix)

### Mask

- mask_mode=absent: 13.4070 us (0.069828 us/matrix)
- mask_mode=full: 13.1150 us (0.068307 us/matrix)
- mask_mode=scalar: 13.1780 us (0.068635 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

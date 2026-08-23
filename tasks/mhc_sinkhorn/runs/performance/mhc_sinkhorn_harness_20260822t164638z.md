# MhcSinkhorn local A3 performance — harness_20260822t164638z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2900 | 13.290000 | 0.0752 | 13.1666, 13.2144, 13.2186, 13.2418, 13.2900, 13.4222, 13.5196, 13.5822, 14.6804 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.1404 | 3.285100 | 0.3044 | 13.0168, 13.0254, 13.0944, 13.1216, 13.1404, 13.1684, 13.2366, 13.2394, 13.3898 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5036 | 0.281325 | 3.5546 | 13.1608, 13.2084, 13.3864, 13.3904, 13.5036, 13.6364, 13.6988, 13.7944, 13.9380 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.7150 | 0.071432 | 13.9993 | 13.4730, 13.4930, 13.6410, 13.6780, 13.7150, 13.7660, 14.1250, 14.5830, 14.6060 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 17.1850 | 0.022376 | 44.6901 | 17.1500, 17.1740, 17.1740, 17.1790, 17.1850, 17.1920, 17.2060, 17.2090, 17.2380 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4496 | 0.070050 | 14.2755 | 12.7836, 13.3540, 13.4108, 13.4280, 13.4496, 13.5484, 13.6020, 13.7660, 14.1140 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4444 | 0.070023 | 14.2810 | 12.9484, 13.1560, 13.2444, 13.3484, 13.4444, 13.5536, 13.5960, 13.6480, 13.8208 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 28.4080 | 0.147958 | 6.7587 | 28.3560, 28.3620, 28.3740, 28.3880, 28.4080, 28.5220, 28.5260, 28.5920, 28.7520 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.2036 | 0.100019 | 9.9981 | 18.9956, 19.1460, 19.1884, 19.2000, 19.2036, 19.2288, 19.2336, 19.2400, 19.2464 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 21.6250 | 0.112630 | 8.8786 | 21.5500, 21.5640, 21.5910, 21.6230, 21.6250, 21.6300, 21.6510, 21.6590, 21.6740 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.6970 | 0.071339 | 14.0177 | 13.1520, 13.1840, 13.3020, 13.5340, 13.6970, 13.8070, 14.1500, 14.3670, 14.5420 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.3660 | 0.069615 | 14.3648 | 12.9670, 13.1290, 13.1610, 13.2810, 13.3660, 13.8830, 13.9940, 14.3360, 14.8330 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.4970 | 0.070297 | 14.2254 | 13.1350, 13.2100, 13.3380, 13.4910, 13.4970, 14.3590, 14.3650, 14.6950, 15.2900 |

## Matrix-count amortization

- matrix_count=1: 13.2900 us (13.290000 us/matrix)
- matrix_count=4: 13.1404 us (3.285100 us/matrix)
- matrix_count=48: 13.5036 us (0.281325 us/matrix)
- matrix_count=192: 13.7150 us (0.071432 us/matrix)
- matrix_count=768: 17.1850 us (0.022376 us/matrix)

## Iteration cost model

- iterations=1: 13.4496 us (0.070050 us/matrix)
- iterations=5: 13.4444 us (0.070023 us/matrix)
- iterations=20: 13.7150 us (0.071432 us/matrix)
- iterations=100: 28.4080 us (0.147958 us/matrix)

Least-squares proxy: `T(iterations) = 12.2698 us + iterations * 0.1582 us`, R²=0.974408.

## Orthogonal comparisons

### N

- n=4: 19.2036 us (0.100019 us/matrix)
- n=6: 21.6250 us (0.112630 us/matrix)
- n=8: 13.7150 us (0.071432 us/matrix)

### DType

- dtype=float16: 13.6970 us (0.071339 us/matrix)
- dtype=float32: 13.7150 us (0.071432 us/matrix)

### Mask

- mask_mode=absent: 13.7150 us (0.071432 us/matrix)
- mask_mode=full: 13.4970 us (0.070297 us/matrix)
- mask_mode=scalar: 13.3660 us (0.069615 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

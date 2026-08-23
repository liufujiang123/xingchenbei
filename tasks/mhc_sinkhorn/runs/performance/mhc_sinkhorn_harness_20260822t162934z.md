# MhcSinkhorn local A3 performance — harness_20260822t162934z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.3104 | 13.310400 | 0.0751 | 12.9134, 13.0672, 13.1272, 13.1958, 13.3104, 13.4486, 13.5622, 13.7152, 13.8004 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0214 | 3.255350 | 0.3072 | 12.7752, 12.9188, 12.9368, 13.0032, 13.0214, 13.0250, 13.0488, 13.2104, 13.2538 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.6764 | 0.284925 | 3.5097 | 13.4960, 13.5364, 13.5480, 13.6416, 13.6764, 13.6988, 13.7876, 13.8328, 13.8344 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 14.2850 | 0.074401 | 13.4407 | 13.9720, 14.0240, 14.0420, 14.1460, 14.2850, 14.5860, 14.6500, 14.8520, 14.8830 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 33.6950 | 0.043874 | 22.7927 | 33.6570, 33.6780, 33.6790, 33.6850, 33.6950, 33.7070, 33.7180, 33.7200, 33.7240 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.6748 | 0.071223 | 14.0404 | 13.1740, 13.5204, 13.5736, 13.6340, 13.6748, 13.7988, 13.8232, 13.9328, 14.0400 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4464 | 0.070033 | 14.2789 | 13.1704, 13.2808, 13.3484, 13.3624, 13.4464, 13.5604, 13.6208, 13.6272, 13.7352 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 39.4240 | 0.205333 | 4.8701 | 39.3880, 39.3980, 39.4140, 39.4180, 39.4240, 39.4300, 39.4320, 39.4580, 40.5680 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 18.8096 | 0.097967 | 10.2076 | 18.7828, 18.7904, 18.8016, 18.8020, 18.8096, 18.8128, 18.8408, 18.8556, 19.0396 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 20.9790 | 0.109266 | 9.1520 | 20.9380, 20.9490, 20.9690, 20.9710, 20.9790, 21.0040, 21.0120, 21.1180, 21.1600 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 14.2390 | 0.074161 | 13.4841 | 13.5610, 13.6030, 13.9340, 13.9960, 14.2390, 14.3870, 14.4130, 14.7310, 14.8620 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.8430 | 0.072099 | 13.8698 | 13.5670, 13.6210, 13.6370, 13.7860, 13.8430, 14.5050, 14.5170, 14.5390, 14.5630 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.9040 | 0.072417 | 13.8090 | 13.2820, 13.3030, 13.3720, 13.7230, 13.9040, 13.9320, 14.0480, 14.0580, 14.2070 |

## Matrix-count amortization

- matrix_count=1: 13.3104 us (13.310400 us/matrix)
- matrix_count=4: 13.0214 us (3.255350 us/matrix)
- matrix_count=48: 13.6764 us (0.284925 us/matrix)
- matrix_count=192: 14.2850 us (0.074401 us/matrix)
- matrix_count=768: 33.6950 us (0.043874 us/matrix)

## Iteration cost model

- iterations=1: 13.6748 us (0.071223 us/matrix)
- iterations=5: 13.4464 us (0.070033 us/matrix)
- iterations=20: 14.2850 us (0.074401 us/matrix)
- iterations=100: 39.4240 us (0.205333 us/matrix)

Least-squares proxy: `T(iterations) = 11.6076 us + iterations * 0.2730 us`, R²=0.976756.

## Orthogonal comparisons

### N

- n=4: 18.8096 us (0.097967 us/matrix)
- n=6: 20.9790 us (0.109266 us/matrix)
- n=8: 14.2850 us (0.074401 us/matrix)

### DType

- dtype=float16: 14.2390 us (0.074161 us/matrix)
- dtype=float32: 14.2850 us (0.074401 us/matrix)

### Mask

- mask_mode=absent: 14.2850 us (0.074401 us/matrix)
- mask_mode=full: 13.9040 us (0.072417 us/matrix)
- mask_mode=scalar: 13.8430 us (0.072099 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

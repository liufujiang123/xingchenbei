# MhcSinkhorn retained baseline — harness_20260822t160254z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2354 | 13.235400 | 0.0756 | 13.0376, 13.0784, 13.0894, 13.2058, 13.2354, 13.2454, 13.3372, 13.3638, 13.7224 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.9006 | 3.225150 | 0.3101 | 12.8322, 12.8404, 12.8584, 12.8618, 12.9006, 12.9512, 12.9782, 13.0526, 13.2004 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.2668 | 0.276392 | 3.6181 | 12.7748, 13.0932, 13.1520, 13.2040, 13.2668, 13.3612, 13.4580, 13.5860, 13.7544 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 22.9580 | 0.119573 | 8.3631 | 22.9340, 22.9440, 22.9440, 22.9470, 22.9580, 22.9590, 22.9610, 22.9670, 23.0240 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 75.9170 | 0.098850 | 10.1163 | 75.8850, 75.8960, 75.9100, 75.9120, 75.9170, 75.9200, 75.9310, 75.9380, 76.5960 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4928 | 0.070275 | 14.2298 | 12.9232, 13.3624, 13.4016, 13.4920, 13.4928, 13.5252, 13.5480, 13.6184, 13.6860 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.2932 | 0.069235 | 14.4435 | 12.8892, 13.1372, 13.2564, 13.2816, 13.2932, 13.3108, 13.3196, 13.3800, 13.3944 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 91.8820 | 0.478552 | 2.0896 | 91.8580, 91.8620, 91.8700, 91.8720, 91.8820, 91.8860, 91.8900, 91.9340, 93.0200 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 30.0468 | 0.156494 | 6.3900 | 30.0360, 30.0428, 30.0464, 30.0464, 30.0468, 30.0480, 30.0820, 30.2680, 30.3496 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 41.1480 | 0.214313 | 4.6661 | 41.1260, 41.1400, 41.1440, 41.1450, 41.1480, 41.1490, 41.1510, 41.1570, 41.1770 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 22.9460 | 0.119510 | 8.3675 | 22.9150, 22.9250, 22.9410, 22.9460, 22.9460, 22.9490, 22.9600, 22.9720, 22.9950 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 23.6780 | 0.123323 | 8.1088 | 23.0130, 23.0270, 23.0290, 23.6670, 23.6780, 23.6830, 23.6850, 23.6960, 23.6980 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 23.4830 | 0.122307 | 8.1761 | 23.4480, 23.4480, 23.4580, 23.4640, 23.4830, 23.4900, 23.4950, 23.5020, 23.5040 |

## Matrix-count amortization

- matrix_count=1: 13.2354 us (13.235400 us/matrix)
- matrix_count=4: 12.9006 us (3.225150 us/matrix)
- matrix_count=48: 13.2668 us (0.276392 us/matrix)
- matrix_count=192: 22.9580 us (0.119573 us/matrix)
- matrix_count=768: 75.9170 us (0.098850 us/matrix)

## Iteration cost model

- iterations=1: 13.4928 us (0.070275 us/matrix)
- iterations=5: 13.2932 us (0.069235 us/matrix)
- iterations=20: 22.9580 us (0.119573 us/matrix)
- iterations=100: 91.8820 us (0.478552 us/matrix)

Least-squares proxy: `T(iterations) = 9.7162 us + iterations * 0.8156 us`, R²=0.995639.

## Orthogonal comparisons

### N

- n=4: 30.0468 us (0.156494 us/matrix)
- n=6: 41.1480 us (0.214313 us/matrix)
- n=8: 22.9580 us (0.119573 us/matrix)

### DType

- dtype=float16: 22.9460 us (0.119510 us/matrix)
- dtype=float32: 22.9580 us (0.119573 us/matrix)

### Mask

- mask_mode=absent: 22.9580 us (0.119573 us/matrix)
- mask_mode=full: 23.4830 us (0.122307 us/matrix)
- mask_mode=scalar: 23.6780 us (0.123323 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

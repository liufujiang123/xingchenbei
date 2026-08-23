# MhcSinkhorn local A3 performance — harness_20260822t170136z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 12.9326 | 12.932600 | 0.0773 | 12.6648, 12.8340, 12.8708, 12.9136, 12.9326, 12.9920, 13.1008, 13.3438, 13.3610 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.7302 | 3.182550 | 0.3142 | 12.5872, 12.5960, 12.6326, 12.6552, 12.7302, 12.7466, 12.8350, 12.9706, 13.1086 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 12.9864 | 0.270550 | 3.6962 | 12.9228, 12.9344, 12.9492, 12.9520, 12.9864, 13.0000, 13.0172, 13.0308, 13.2316 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 12.8540 | 0.066948 | 14.9370 | 12.6490, 12.6500, 12.7070, 12.7290, 12.8540, 13.1960, 13.6730, 13.8390, 14.0190 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 16.9310 | 0.022046 | 45.3606 | 16.8920, 16.9040, 16.9080, 16.9280, 16.9310, 16.9340, 16.9460, 16.9830, 17.6070 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.0836 | 0.068144 | 14.6749 | 12.5356, 12.9360, 12.9996, 13.0660, 13.0836, 13.1000, 13.1360, 13.2168, 13.4220 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 12.8284 | 0.066815 | 14.9668 | 12.4788, 12.7156, 12.7604, 12.7916, 12.8284, 12.9804, 12.9976, 13.1372, 13.1568 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.4740 | 0.143094 | 6.9884 | 27.4540, 27.4560, 27.4600, 27.4680, 27.4740, 27.4820, 27.4840, 27.4920, 28.4320 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 17.2976 | 0.090092 | 11.0998 | 16.9976, 17.1664, 17.2664, 17.2912, 17.2976, 17.2988, 17.3176, 17.3316, 17.3380 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.5640 | 0.101896 | 9.8139 | 18.8760, 18.8790, 19.5580, 19.5600, 19.5640, 19.5700, 19.5810, 19.5840, 19.5910 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 12.9710 | 0.067557 | 14.8023 | 12.7240, 12.7580, 12.8880, 12.9420, 12.9710, 13.2060, 13.7760, 13.8990, 13.9730 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 12.9970 | 0.067693 | 14.7726 | 12.5420, 12.6670, 12.6930, 12.7080, 12.9970, 13.4000, 13.6660, 13.7440, 14.0690 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.5260 | 0.070448 | 14.1949 | 12.7560, 12.7850, 13.0070, 13.1420, 13.5260, 13.8230, 13.9750, 14.0930, 14.5180 |

## Matrix-count amortization

- matrix_count=1: 12.9326 us (12.932600 us/matrix)
- matrix_count=4: 12.7302 us (3.182550 us/matrix)
- matrix_count=48: 12.9864 us (0.270550 us/matrix)
- matrix_count=192: 12.8540 us (0.066948 us/matrix)
- matrix_count=768: 16.9310 us (0.022046 us/matrix)

## Iteration cost model

- iterations=1: 13.0836 us (0.068144 us/matrix)
- iterations=5: 12.8284 us (0.066815 us/matrix)
- iterations=20: 12.8540 us (0.066948 us/matrix)
- iterations=100: 27.4740 us (0.143094 us/matrix)

Least-squares proxy: `T(iterations) = 11.7053 us + iterations * 0.1541 us`, R²=0.965446.

## Orthogonal comparisons

### N

- n=4: 17.2976 us (0.090092 us/matrix)
- n=6: 19.5640 us (0.101896 us/matrix)
- n=8: 12.8540 us (0.066948 us/matrix)

### DType

- dtype=float16: 12.9710 us (0.067557 us/matrix)
- dtype=float32: 12.8540 us (0.066948 us/matrix)

### Mask

- mask_mode=absent: 12.8540 us (0.066948 us/matrix)
- mask_mode=full: 13.5260 us (0.070448 us/matrix)
- mask_mode=scalar: 12.9970 us (0.067693 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

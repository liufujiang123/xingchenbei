# MhcSinkhorn local A3 performance — harness_20260822t165856z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2942 | 13.294200 | 0.0752 | 13.0870, 13.1284, 13.2536, 13.2718, 13.2942, 13.2966, 13.3508, 13.3746, 13.3878 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.2690 | 3.317250 | 0.3015 | 13.1440, 13.1520, 13.1858, 13.2114, 13.2690, 13.2976, 13.2998, 13.3804, 13.4574 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5548 | 0.282392 | 3.5412 | 13.1292, 13.3088, 13.3792, 13.5196, 13.5548, 13.5732, 13.6160, 13.6512, 13.6844 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.8290 | 0.072026 | 13.8839 | 13.4990, 13.5010, 13.5230, 13.6000, 13.8290, 13.8950, 14.0220, 14.4440, 14.5320 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 22.8120 | 0.029703 | 33.6665 | 22.7550, 22.7610, 22.7640, 22.8070, 22.8120, 22.8150, 22.8410, 22.8550, 22.8570 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.5216 | 0.070425 | 14.1995 | 12.8468, 13.3220, 13.4212, 13.5108, 13.5216, 13.5532, 13.6520, 13.7012, 13.7952 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.2684 | 0.069106 | 14.4705 | 12.8472, 13.0276, 13.2028, 13.2076, 13.2684, 13.3244, 13.3716, 13.4280, 13.4816 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.5720 | 0.143604 | 6.9636 | 27.4240, 27.4520, 27.4600, 27.4860, 27.5720, 27.6200, 27.6280, 27.6560, 27.6940 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.7312 | 0.087142 | 11.4756 | 16.6848, 16.7088, 16.7112, 16.7164, 16.7312, 16.7456, 16.7476, 16.7624, 16.7940 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.5860 | 0.096802 | 10.3304 | 18.4930, 18.5590, 18.5780, 18.5820, 18.5860, 18.5910, 18.6090, 18.6300, 18.6790 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.7210 | 0.071464 | 13.9931 | 13.2810, 13.4980, 13.6130, 13.7200, 13.7210, 14.0130, 14.2440, 14.5820, 14.8840 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.6990 | 0.071349 | 14.0156 | 13.1920, 13.4490, 13.6880, 13.6910, 13.6990, 14.1400, 14.2160, 14.3440, 15.5010 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 14.0450 | 0.073151 | 13.6703 | 13.3880, 13.3960, 13.5620, 13.7240, 14.0450, 14.1380, 14.2960, 14.4130, 14.4760 |

## Matrix-count amortization

- matrix_count=1: 13.2942 us (13.294200 us/matrix)
- matrix_count=4: 13.2690 us (3.317250 us/matrix)
- matrix_count=48: 13.5548 us (0.282392 us/matrix)
- matrix_count=192: 13.8290 us (0.072026 us/matrix)
- matrix_count=768: 22.8120 us (0.029703 us/matrix)

## Iteration cost model

- iterations=1: 13.5216 us (0.070425 us/matrix)
- iterations=5: 13.2684 us (0.069106 us/matrix)
- iterations=20: 13.8290 us (0.072026 us/matrix)
- iterations=100: 27.5720 us (0.143604 us/matrix)

Least-squares proxy: `T(iterations) = 12.3370 us + iterations * 0.1495 us`, R²=0.976796.

## Orthogonal comparisons

### N

- n=4: 16.7312 us (0.087142 us/matrix)
- n=6: 18.5860 us (0.096802 us/matrix)
- n=8: 13.8290 us (0.072026 us/matrix)

### DType

- dtype=float16: 13.7210 us (0.071464 us/matrix)
- dtype=float32: 13.8290 us (0.072026 us/matrix)

### Mask

- mask_mode=absent: 13.8290 us (0.072026 us/matrix)
- mask_mode=full: 14.0450 us (0.073151 us/matrix)
- mask_mode=scalar: 13.6990 us (0.071349 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

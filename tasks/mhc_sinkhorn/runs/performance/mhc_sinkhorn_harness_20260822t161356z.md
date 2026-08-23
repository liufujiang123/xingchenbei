# MhcSinkhorn local A3 performance — harness_20260822t161356z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2144 | 13.214400 | 0.0757 | 12.9796, 13.0366, 13.0430, 13.1152, 13.2144, 13.3456, 13.3530, 13.3752, 13.8216 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0896 | 3.272400 | 0.3056 | 12.8592, 12.9442, 13.0634, 13.0778, 13.0896, 13.0896, 13.1088, 13.1350, 13.2030 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5260 | 0.281792 | 3.5487 | 13.2508, 13.3392, 13.4156, 13.5228, 13.5260, 13.6036, 13.6128, 13.7384, 13.7776 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.6990 | 0.071349 | 14.0156 | 13.4110, 13.4160, 13.6240, 13.6280, 13.6990, 13.8150, 13.8900, 14.1950, 14.3050 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 37.4770 | 0.048798 | 20.4926 | 37.4670, 37.4670, 37.4700, 37.4720, 37.4770, 37.4810, 37.4860, 37.5010, 37.5480 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.6668 | 0.071181 | 14.0486 | 13.2872, 13.4856, 13.5092, 13.5244, 13.6668, 13.7900, 13.9472, 13.9576, 14.3652 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.3892 | 0.069735 | 14.3399 | 12.9248, 13.2260, 13.3240, 13.3332, 13.3892, 13.4596, 13.5316, 13.7272, 14.0076 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 44.7980 | 0.233323 | 4.2859 | 44.7660, 44.7800, 44.7840, 44.7900, 44.7980, 44.8060, 44.8140, 44.8260, 45.9260 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 22.4160 | 0.116750 | 8.5653 | 22.3800, 22.3856, 22.3928, 22.3936, 22.4160, 22.4184, 22.4224, 22.4284, 22.4520 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 27.2870 | 0.142120 | 7.0363 | 27.2640, 27.2660, 27.2760, 27.2820, 27.2870, 27.2930, 27.2960, 27.3220, 27.3300 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.5880 | 0.070771 | 14.1301 | 13.3700, 13.4460, 13.5240, 13.5540, 13.5880, 13.7930, 14.1120, 14.2850, 14.7410 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.6340 | 0.071010 | 14.0824 | 13.5790, 13.6000, 13.6060, 13.6300, 13.6340, 13.6940, 14.2370, 14.5580, 15.1680 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.6540 | 0.071115 | 14.0618 | 13.5450, 13.5610, 13.6080, 13.6360, 13.6540, 13.9760, 14.2400, 14.4770, 15.4880 |

## Matrix-count amortization

- matrix_count=1: 13.2144 us (13.214400 us/matrix)
- matrix_count=4: 13.0896 us (3.272400 us/matrix)
- matrix_count=48: 13.5260 us (0.281792 us/matrix)
- matrix_count=192: 13.6990 us (0.071349 us/matrix)
- matrix_count=768: 37.4770 us (0.048798 us/matrix)

## Iteration cost model

- iterations=1: 13.6668 us (0.071181 us/matrix)
- iterations=5: 13.3892 us (0.069735 us/matrix)
- iterations=20: 13.6990 us (0.071349 us/matrix)
- iterations=100: 44.7980 us (0.233323 us/matrix)

Least-squares proxy: `T(iterations) = 10.9510 us + iterations * 0.3313 us`, R²=0.970099.

## Orthogonal comparisons

### N

- n=4: 22.4160 us (0.116750 us/matrix)
- n=6: 27.2870 us (0.142120 us/matrix)
- n=8: 13.6990 us (0.071349 us/matrix)

### DType

- dtype=float16: 13.5880 us (0.070771 us/matrix)
- dtype=float32: 13.6990 us (0.071349 us/matrix)

### Mask

- mask_mode=absent: 13.6990 us (0.071349 us/matrix)
- mask_mode=full: 13.6540 us (0.071115 us/matrix)
- mask_mode=scalar: 13.6340 us (0.071010 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

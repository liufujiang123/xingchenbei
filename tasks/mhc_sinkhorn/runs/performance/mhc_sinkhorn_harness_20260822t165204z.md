# MhcSinkhorn local A3 performance — harness_20260822t165204z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.3606 | 13.360600 | 0.0748 | 13.1136, 13.2150, 13.2918, 13.3440, 13.3606, 13.3832, 13.4332, 13.4708, 13.7714 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.1854 | 3.296350 | 0.3034 | 13.0578, 13.0710, 13.1152, 13.1462, 13.1854, 13.2260, 13.2366, 13.2740, 13.6500 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5572 | 0.282442 | 3.5406 | 13.3164, 13.3880, 13.4872, 13.5108, 13.5572, 13.5748, 13.5960, 13.6756, 13.7244 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.7800 | 0.071771 | 13.9332 | 13.1750, 13.2910, 13.3130, 13.6580, 13.7800, 14.0000, 14.2780, 14.3930, 14.4550 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 22.8010 | 0.029689 | 33.6827 | 22.3830, 22.7730, 22.7810, 22.7850, 22.8010, 22.8040, 22.8040, 22.8120, 23.1000 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.8312 | 0.072037 | 13.8817 | 13.5268, 13.6540, 13.7824, 13.8016, 13.8312, 13.8784, 13.9064, 13.9944, 14.0144 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.6140 | 0.070906 | 14.1031 | 13.2972, 13.4660, 13.5900, 13.5924, 13.6140, 13.6364, 13.6536, 13.7196, 13.7412 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.4860 | 0.143156 | 6.9854 | 27.4560, 27.4600, 27.4740, 27.4760, 27.4860, 27.4900, 27.5080, 27.5120, 28.4380 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.4608 | 0.085733 | 11.6641 | 16.4000, 16.4252, 16.4304, 16.4532, 16.4608, 16.4620, 16.4836, 16.4864, 16.5192 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.9090 | 0.098484 | 10.1539 | 18.8800, 18.8810, 18.8980, 18.9090, 18.9090, 18.9130, 18.9150, 18.9180, 19.0300 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.5270 | 0.070453 | 14.1938 | 13.1580, 13.4140, 13.5190, 13.5230, 13.5270, 13.6940, 14.3570, 14.4320, 14.4320 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 14.0110 | 0.072974 | 13.7035 | 13.4830, 13.5690, 13.6160, 13.7980, 14.0110, 14.2720, 14.2780, 14.6270, 14.7490 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 14.1110 | 0.073495 | 13.6064 | 13.2920, 13.4220, 13.5820, 13.7350, 14.1110, 14.1850, 14.2080, 14.3350, 14.3740 |

## Matrix-count amortization

- matrix_count=1: 13.3606 us (13.360600 us/matrix)
- matrix_count=4: 13.1854 us (3.296350 us/matrix)
- matrix_count=48: 13.5572 us (0.282442 us/matrix)
- matrix_count=192: 13.7800 us (0.071771 us/matrix)
- matrix_count=768: 22.8010 us (0.029689 us/matrix)

## Iteration cost model

- iterations=1: 13.8312 us (0.072037 us/matrix)
- iterations=5: 13.6140 us (0.070906 us/matrix)
- iterations=20: 13.7800 us (0.071771 us/matrix)
- iterations=100: 27.4860 us (0.143156 us/matrix)

Least-squares proxy: `T(iterations) = 12.5838 us + iterations * 0.1458 us`, R²=0.969191.

## Orthogonal comparisons

### N

- n=4: 16.4608 us (0.085733 us/matrix)
- n=6: 18.9090 us (0.098484 us/matrix)
- n=8: 13.7800 us (0.071771 us/matrix)

### DType

- dtype=float16: 13.5270 us (0.070453 us/matrix)
- dtype=float32: 13.7800 us (0.071771 us/matrix)

### Mask

- mask_mode=absent: 13.7800 us (0.071771 us/matrix)
- mask_mode=full: 14.1110 us (0.073495 us/matrix)
- mask_mode=scalar: 14.0110 us (0.072974 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

# MhcSinkhorn local A3 performance — n8_coldiv_mask_hoist_v1

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.1688 | 13.168800 | 0.0759 | 12.9124, 13.0138, 13.0382, 13.0422, 13.1688, 13.2468, 13.2946, 13.4132, 14.0434 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.9354 | 3.233850 | 0.3092 | 12.7834, 12.7998, 12.8470, 12.8784, 12.9354, 13.0208, 13.0454, 13.1590, 13.1622 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.2244 | 0.275508 | 3.6297 | 12.9280, 13.0040, 13.0972, 13.1144, 13.2244, 13.4012, 13.4308, 13.4944, 13.7224 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.2180 | 0.068844 | 14.5256 | 12.7590, 12.8250, 13.1890, 13.1910, 13.2180, 13.2700, 13.3550, 13.4150, 14.4220 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 17.3330 | 0.022569 | 44.3085 | 17.2910, 17.2990, 17.3090, 17.3190, 17.3330, 17.3410, 17.3430, 17.3460, 17.3620 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.3520 | 0.069542 | 14.3799 | 12.7172, 13.0296, 13.2480, 13.3216, 13.3520, 13.3936, 13.4436, 13.4772, 13.9176 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.3720 | 0.069646 | 14.3584 | 12.9092, 12.9172, 13.1688, 13.1700, 13.3720, 13.4108, 13.4268, 13.4280, 13.5296 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 28.6600 | 0.149271 | 6.6992 | 27.2520, 28.6120, 28.6420, 28.6420, 28.6600, 28.7060, 28.7060, 28.7540, 29.6060 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 17.0700 | 0.088906 | 11.2478 | 16.7628, 16.7700, 16.7776, 16.8104, 17.0700, 17.0888, 17.1184, 17.1380, 17.1604 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.7920 | 0.097875 | 10.2171 | 18.7080, 18.7360, 18.7520, 18.7750, 18.7920, 18.8040, 18.8110, 18.8170, 18.8730 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.3760 | 0.069667 | 14.3541 | 13.0590, 13.0750, 13.1130, 13.3730, 13.3760, 13.5450, 13.7030, 14.1450, 14.3260 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.3170 | 0.069359 | 14.4177 | 12.9890, 13.1000, 13.2700, 13.2830, 13.3170, 13.3330, 13.8540, 14.4580, 15.1700 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.3140 | 0.069344 | 14.4209 | 13.1910, 13.1960, 13.2030, 13.2620, 13.3140, 13.6500, 13.7920, 13.8000, 14.5740 |

## Matrix-count amortization

- matrix_count=1: 13.1688 us (13.168800 us/matrix)
- matrix_count=4: 12.9354 us (3.233850 us/matrix)
- matrix_count=48: 13.2244 us (0.275508 us/matrix)
- matrix_count=192: 13.2180 us (0.068844 us/matrix)
- matrix_count=768: 17.3330 us (0.022569 us/matrix)

## Iteration cost model

- iterations=1: 13.3520 us (0.069542 us/matrix)
- iterations=5: 13.3720 us (0.069646 us/matrix)
- iterations=20: 13.2180 us (0.068844 us/matrix)
- iterations=100: 28.6600 us (0.149271 us/matrix)

Least-squares proxy: `T(iterations) = 12.0301 us + iterations * 0.1626 us`, R²=0.965913.

## Orthogonal comparisons

### N

- n=4: 17.0700 us (0.088906 us/matrix)
- n=6: 18.7920 us (0.097875 us/matrix)
- n=8: 13.2180 us (0.068844 us/matrix)

### DType

- dtype=float16: 13.3760 us (0.069667 us/matrix)
- dtype=float32: 13.2180 us (0.068844 us/matrix)

### Mask

- mask_mode=absent: 13.2180 us (0.068844 us/matrix)
- mask_mode=full: 13.3140 us (0.069344 us/matrix)
- mask_mode=scalar: 13.3170 us (0.069359 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

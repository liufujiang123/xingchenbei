# MhcSinkhorn local A3 performance — harness_20260822t162350z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.4022 | 13.402200 | 0.0746 | 13.2486, 13.3532, 13.3928, 13.4014, 13.4022, 13.5312, 13.5438, 13.8980, 14.4646 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.4332 | 3.358300 | 0.2978 | 13.1204, 13.2032, 13.3688, 13.4162, 13.4332, 13.4504, 13.4798, 13.6794, 14.3114 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.6380 | 0.284125 | 3.5196 | 13.4028, 13.4784, 13.5500, 13.5852, 13.6380, 13.6476, 13.6492, 13.6816, 13.8068 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.8330 | 0.072047 | 13.8799 | 13.5260, 13.6680, 13.6840, 13.7570, 13.8330, 14.2110, 14.2600, 14.6070, 14.8160 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 33.4170 | 0.043512 | 22.9823 | 33.3720, 33.3860, 33.3970, 33.4010, 33.4170, 33.4240, 33.4590, 33.5980, 33.7060 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.7008 | 0.071358 | 14.0138 | 13.1084, 13.3448, 13.6060, 13.6616, 13.7008, 13.7140, 13.8672, 13.9864, 14.2096 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.7660 | 0.071698 | 13.9474 | 13.3816, 13.4916, 13.5688, 13.7124, 13.7660, 13.7776, 13.7884, 13.8740, 13.9228 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 40.2220 | 0.209490 | 4.7735 | 40.1760, 40.1820, 40.1940, 40.2080, 40.2220, 40.2320, 40.2340, 40.2380, 41.2220 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.0848 | 0.099400 | 10.0604 | 19.0600, 19.0640, 19.0652, 19.0772, 19.0848, 19.0856, 19.1156, 19.1276, 19.1456 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 21.3910 | 0.111411 | 8.9757 | 21.3730, 21.3820, 21.3820, 21.3870, 21.3910, 21.3920, 21.4050, 21.4080, 21.4270 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 14.4760 | 0.075396 | 13.2633 | 13.6930, 13.7210, 13.8110, 13.8190, 14.4760, 14.4850, 14.5420, 14.7960, 14.9960 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.8010 | 0.071880 | 13.9120 | 13.4410, 13.4450, 13.5410, 13.6710, 13.8010, 13.8450, 14.0670, 14.1820, 14.2940 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.8700 | 0.072240 | 13.8428 | 13.5280, 13.5440, 13.6690, 13.7990, 13.8700, 13.9760, 14.3860, 14.3920, 14.5420 |

## Matrix-count amortization

- matrix_count=1: 13.4022 us (13.402200 us/matrix)
- matrix_count=4: 13.4332 us (3.358300 us/matrix)
- matrix_count=48: 13.6380 us (0.284125 us/matrix)
- matrix_count=192: 13.8330 us (0.072047 us/matrix)
- matrix_count=768: 33.4170 us (0.043512 us/matrix)

## Iteration cost model

- iterations=1: 13.7008 us (0.071358 us/matrix)
- iterations=5: 13.7660 us (0.071698 us/matrix)
- iterations=20: 13.8330 us (0.072047 us/matrix)
- iterations=100: 40.2220 us (0.209490 us/matrix)

Least-squares proxy: `T(iterations) = 11.5336 us + iterations * 0.2809 us`, R²=0.970253.

## Orthogonal comparisons

### N

- n=4: 19.0848 us (0.099400 us/matrix)
- n=6: 21.3910 us (0.111411 us/matrix)
- n=8: 13.8330 us (0.072047 us/matrix)

### DType

- dtype=float16: 14.4760 us (0.075396 us/matrix)
- dtype=float32: 13.8330 us (0.072047 us/matrix)

### Mask

- mask_mode=absent: 13.8330 us (0.072047 us/matrix)
- mask_mode=full: 13.8700 us (0.072240 us/matrix)
- mask_mode=scalar: 13.8010 us (0.071880 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

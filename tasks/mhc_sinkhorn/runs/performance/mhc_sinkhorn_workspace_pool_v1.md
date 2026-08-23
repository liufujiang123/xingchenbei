# MhcSinkhorn local A3 performance — workspace_pool_v1

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.1718 | 13.171800 | 0.0759 | 12.9276, 12.9582, 12.9834, 13.1612, 13.1718, 13.1968, 13.2182, 13.4846, 14.2208 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.0122 | 3.253050 | 0.3074 | 12.8688, 12.9016, 12.9264, 12.9662, 13.0122, 13.0396, 13.0610, 13.1292, 13.3026 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.4256 | 0.279700 | 3.5753 | 13.3356, 13.3708, 13.4088, 13.4140, 13.4256, 13.4348, 13.5104, 13.7436, 13.8564 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.6480 | 0.071083 | 14.0680 | 13.4330, 13.4800, 13.5210, 13.5920, 13.6480, 13.6860, 13.7250, 13.9710, 14.2150 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 17.0980 | 0.022263 | 44.9175 | 17.0160, 17.0780, 17.0890, 17.0930, 17.0980, 17.1270, 17.1400, 17.1870, 17.4130 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.5444 | 0.070544 | 14.1756 | 12.9468, 13.2084, 13.4552, 13.5228, 13.5444, 13.5900, 13.7120, 13.7228, 13.7844 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.4220 | 0.069906 | 14.3049 | 13.1768, 13.2664, 13.3584, 13.3920, 13.4220, 13.4404, 13.5512, 13.6332, 13.7668 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 28.1620 | 0.146677 | 6.8177 | 27.8300, 27.8320, 27.8780, 27.8860, 28.1620, 28.1820, 28.2760, 28.3800, 29.1520 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.5300 | 0.086094 | 11.6152 | 16.4948, 16.4952, 16.5104, 16.5216, 16.5300, 16.5380, 16.5484, 16.5608, 16.6036 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.9050 | 0.098464 | 10.1560 | 18.8380, 18.8450, 18.8690, 18.8690, 18.9050, 18.9880, 19.0910, 19.1070, 19.3900 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.5050 | 0.070339 | 14.2170 | 13.2000, 13.2180, 13.2510, 13.4400, 13.5050, 13.7650, 14.1940, 14.2220, 14.4060 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.3830 | 0.069703 | 14.3466 | 13.2270, 13.3410, 13.3680, 13.3770, 13.3830, 13.5250, 13.7970, 13.8720, 14.1340 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.5540 | 0.070594 | 14.1656 | 13.1830, 13.4860, 13.5340, 13.5340, 13.5540, 13.6990, 14.0900, 14.4130, 14.9140 |

## Matrix-count amortization

- matrix_count=1: 13.1718 us (13.171800 us/matrix)
- matrix_count=4: 13.0122 us (3.253050 us/matrix)
- matrix_count=48: 13.4256 us (0.279700 us/matrix)
- matrix_count=192: 13.6480 us (0.071083 us/matrix)
- matrix_count=768: 17.0980 us (0.022263 us/matrix)

## Iteration cost model

- iterations=1: 13.5444 us (0.070544 us/matrix)
- iterations=5: 13.4220 us (0.069906 us/matrix)
- iterations=20: 13.6480 us (0.071083 us/matrix)
- iterations=100: 28.1620 us (0.146677 us/matrix)

Least-squares proxy: `T(iterations) = 12.2993 us + iterations * 0.1554 us`, R²=0.971910.

## Orthogonal comparisons

### N

- n=4: 16.5300 us (0.086094 us/matrix)
- n=6: 18.9050 us (0.098464 us/matrix)
- n=8: 13.6480 us (0.071083 us/matrix)

### DType

- dtype=float16: 13.5050 us (0.070339 us/matrix)
- dtype=float32: 13.6480 us (0.071083 us/matrix)

### Mask

- mask_mode=absent: 13.6480 us (0.071083 us/matrix)
- mask_mode=full: 13.5540 us (0.070594 us/matrix)
- mask_mode=scalar: 13.3830 us (0.069703 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

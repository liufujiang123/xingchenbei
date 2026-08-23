# MhcSinkhorn local A3 performance — harness_20260822t165358z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.5080 | 13.508000 | 0.0740 | 13.3058, 13.4548, 13.5014, 13.5018, 13.5080, 13.5604, 13.7054, 13.7142, 14.2476 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.4244 | 3.356100 | 0.2980 | 13.1970, 13.2644, 13.3120, 13.3444, 13.4244, 13.4948, 13.4976, 13.5950, 13.6322 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.7288 | 0.286017 | 3.4963 | 13.4832, 13.6960, 13.7072, 13.7160, 13.7288, 13.8024, 13.9436, 13.9972, 14.0544 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.9590 | 0.072703 | 13.7546 | 13.3200, 13.7740, 13.8120, 13.8840, 13.9590, 14.0140, 14.1700, 14.2790, 14.5660 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 23.2340 | 0.030253 | 33.0550 | 22.8300, 22.8490, 22.9430, 23.2140, 23.2340, 23.2350, 23.2420, 23.2420, 23.2730 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.8648 | 0.072212 | 13.8480 | 13.1920, 13.6436, 13.7072, 13.7808, 13.8648, 13.8748, 13.8880, 14.0360, 14.0448 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.6372 | 0.071027 | 14.0791 | 13.4204, 13.4712, 13.4760, 13.4904, 13.6372, 13.6540, 13.7360, 13.7412, 13.7496 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.8760 | 0.145187 | 6.8876 | 27.0700, 27.8280, 27.8360, 27.8740, 27.8760, 27.8820, 27.8920, 27.9000, 27.9140 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 17.0164 | 0.088627 | 11.2832 | 16.9468, 16.9944, 16.9964, 17.0092, 17.0164, 17.0172, 17.0200, 17.0260, 17.0548 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 19.0990 | 0.099474 | 10.0529 | 18.8860, 18.9110, 19.0660, 19.0890, 19.0990, 19.1020, 19.1070, 19.1130, 19.2520 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.9330 | 0.072568 | 13.7802 | 13.4550, 13.4610, 13.6340, 13.8740, 13.9330, 14.2510, 14.5090, 14.5750, 14.7090 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.9310 | 0.072557 | 13.7822 | 13.6830, 13.7280, 13.8250, 13.8690, 13.9310, 14.5010, 14.5710, 14.5720, 14.6710 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.8300 | 0.072031 | 13.8829 | 13.3790, 13.5740, 13.7340, 13.8280, 13.8300, 14.3180, 14.3940, 14.7330, 14.8630 |

## Matrix-count amortization

- matrix_count=1: 13.5080 us (13.508000 us/matrix)
- matrix_count=4: 13.4244 us (3.356100 us/matrix)
- matrix_count=48: 13.7288 us (0.286017 us/matrix)
- matrix_count=192: 13.9590 us (0.072703 us/matrix)
- matrix_count=768: 23.2340 us (0.030253 us/matrix)

## Iteration cost model

- iterations=1: 13.8648 us (0.072212 us/matrix)
- iterations=5: 13.6372 us (0.071027 us/matrix)
- iterations=20: 13.9590 us (0.072703 us/matrix)
- iterations=100: 27.8760 us (0.145187 us/matrix)

Least-squares proxy: `T(iterations) = 12.6280 us + iterations * 0.1494 us`, R²=0.972396.

## Orthogonal comparisons

### N

- n=4: 17.0164 us (0.088627 us/matrix)
- n=6: 19.0990 us (0.099474 us/matrix)
- n=8: 13.9590 us (0.072703 us/matrix)

### DType

- dtype=float16: 13.9330 us (0.072568 us/matrix)
- dtype=float32: 13.9590 us (0.072703 us/matrix)

### Mask

- mask_mode=absent: 13.9590 us (0.072703 us/matrix)
- mask_mode=full: 13.8300 us (0.072031 us/matrix)
- mask_mode=scalar: 13.9310 us (0.072557 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

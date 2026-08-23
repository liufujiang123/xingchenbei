# MhcSinkhorn local A3 performance — harness_20260822t165201z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.2572 | 13.257200 | 0.0754 | 13.0474, 13.1220, 13.1510, 13.2518, 13.2572, 13.2572, 13.3410, 13.6088, 13.8476 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.1036 | 3.275900 | 0.3053 | 12.9866, 12.9902, 13.0044, 13.0096, 13.1036, 13.1198, 13.1222, 13.1984, 13.8006 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.5712 | 0.282733 | 3.5369 | 13.3332, 13.3516, 13.5244, 13.5376, 13.5712, 13.7632, 13.8324, 13.8908, 14.2704 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.9200 | 0.072500 | 13.7931 | 13.5660, 13.5900, 13.6180, 13.9040, 13.9200, 14.3320, 14.5540, 15.1180, 17.3090 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 23.5310 | 0.030639 | 32.6378 | 22.9190, 23.4780, 23.5060, 23.5200, 23.5310, 23.5320, 23.5350, 23.5530, 23.5620 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4264 | 0.069929 | 14.3002 | 12.9496, 13.3080, 13.3268, 13.3952, 13.4264, 13.4780, 13.4996, 13.6948, 13.9680 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.5692 | 0.070673 | 14.1497 | 13.0684, 13.1408, 13.3904, 13.4724, 13.5692, 13.6500, 13.6528, 13.7444, 13.8172 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 27.0740 | 0.141010 | 7.0917 | 27.0500, 27.0620, 27.0640, 27.0680, 27.0740, 27.1000, 27.1220, 27.9080, 28.3640 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.8084 | 0.087544 | 11.4229 | 16.7680, 16.7740, 16.8036, 16.8056, 16.8084, 16.8088, 16.8172, 16.8796, 16.9568 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.7910 | 0.097870 | 10.2177 | 18.7420, 18.7450, 18.7520, 18.7660, 18.7910, 18.8100, 18.8240, 18.8510, 19.0550 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.7360 | 0.071542 | 13.9779 | 12.7390, 12.8570, 12.8610, 12.9400, 13.7360, 13.8570, 13.8920, 13.9600, 17.1670 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.4520 | 0.070062 | 14.2730 | 12.5190, 12.7810, 13.1650, 13.1890, 13.4520, 13.4960, 13.5390, 13.9740, 14.4380 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.2730 | 0.069130 | 14.4655 | 12.8210, 12.9030, 12.9280, 13.2680, 13.2730, 13.6280, 13.8560, 13.8790, 13.9680 |

## Matrix-count amortization

- matrix_count=1: 13.2572 us (13.257200 us/matrix)
- matrix_count=4: 13.1036 us (3.275900 us/matrix)
- matrix_count=48: 13.5712 us (0.282733 us/matrix)
- matrix_count=192: 13.9200 us (0.072500 us/matrix)
- matrix_count=768: 23.5310 us (0.030639 us/matrix)

## Iteration cost model

- iterations=1: 13.4264 us (0.069929 us/matrix)
- iterations=5: 13.5692 us (0.070673 us/matrix)
- iterations=20: 13.9200 us (0.072500 us/matrix)
- iterations=100: 27.0740 us (0.141010 us/matrix)

Least-squares proxy: `T(iterations) = 12.4829 us + iterations * 0.1433 us`, R²=0.978696.

## Orthogonal comparisons

### N

- n=4: 16.8084 us (0.087544 us/matrix)
- n=6: 18.7910 us (0.097870 us/matrix)
- n=8: 13.9200 us (0.072500 us/matrix)

### DType

- dtype=float16: 13.7360 us (0.071542 us/matrix)
- dtype=float32: 13.9200 us (0.072500 us/matrix)

### Mask

- mask_mode=absent: 13.9200 us (0.072500 us/matrix)
- mask_mode=full: 13.2730 us (0.069130 us/matrix)
- mask_mode=scalar: 13.4520 us (0.070062 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

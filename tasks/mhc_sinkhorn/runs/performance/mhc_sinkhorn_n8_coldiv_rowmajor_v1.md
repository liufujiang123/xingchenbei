# MhcSinkhorn local A3 performance — n8_coldiv_rowmajor_v1

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 12.8358 | 12.835799 | 0.0779 | 12.6734, 12.6988, 12.7930, 12.7934, 12.8358, 12.9880, 13.0174, 13.0726, 13.2800 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.6744 | 3.168600 | 0.3156 | 12.5554, 12.5592, 12.6366, 12.6486, 12.6744, 12.6958, 12.7356, 12.9254, 12.9440 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 12.9136 | 0.269033 | 3.7170 | 12.8348, 12.8396, 12.8772, 12.9048, 12.9136, 12.9512, 13.1272, 13.2792, 13.3372 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 12.9080 | 0.067229 | 14.8745 | 12.5210, 12.7290, 12.8600, 12.8860, 12.9080, 12.9670, 13.0680, 13.3250, 13.3770 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 13.7140 | 0.017857 | 56.0012 | 13.5800, 13.6120, 13.6270, 13.6340, 13.7140, 13.9810, 14.0000, 14.1520, 14.2970 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 12.8988 | 0.067181 | 14.8851 | 12.5204, 12.6116, 12.8012, 12.8492, 12.8988, 12.9520, 12.9560, 13.1492, 13.1672 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.0364 | 0.067898 | 14.7280 | 12.7428, 12.9408, 12.9908, 13.0156, 13.0364, 13.0384, 13.0988, 13.1192, 13.1288 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 33.1460 | 0.172635 | 5.7926 | 33.1080, 33.1320, 33.1320, 33.1400, 33.1460, 33.1560, 33.1680, 34.1260, 34.5360 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 16.6896 | 0.086925 | 11.5042 | 16.6248, 16.6532, 16.6576, 16.6712, 16.6896, 16.6920, 16.6932, 16.6984, 16.8596 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 18.4400 | 0.096042 | 10.4121 | 18.4320, 18.4350, 18.4370, 18.4390, 18.4400, 18.4520, 18.4850, 18.5720, 18.8690 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 12.6630 | 0.065953 | 15.1623 | 12.4890, 12.5960, 12.5970, 12.6240, 12.6630, 12.8930, 13.4320, 13.6340, 13.7780 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 12.8780 | 0.067073 | 14.9091 | 12.6830, 12.7090, 12.8600, 12.8630, 12.8780, 12.9340, 13.4740, 13.5130, 13.8120 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.0240 | 0.067833 | 14.7420 | 12.6830, 12.7260, 12.7630, 13.0160, 13.0240, 13.1120, 13.2630, 13.5150, 14.0800 |

## Matrix-count amortization

- matrix_count=1: 12.8358 us (12.835799 us/matrix)
- matrix_count=4: 12.6744 us (3.168600 us/matrix)
- matrix_count=48: 12.9136 us (0.269033 us/matrix)
- matrix_count=192: 12.9080 us (0.067229 us/matrix)
- matrix_count=768: 13.7140 us (0.017857 us/matrix)

## Iteration cost model

- iterations=1: 12.8988 us (0.067181 us/matrix)
- iterations=5: 13.0364 us (0.067898 us/matrix)
- iterations=20: 12.9080 us (0.067229 us/matrix)
- iterations=100: 33.1460 us (0.172635 us/matrix)

Least-squares proxy: `T(iterations) = 11.2496 us + iterations * 0.2142 us`, R²=0.968325.

## Orthogonal comparisons

### N

- n=4: 16.6896 us (0.086925 us/matrix)
- n=6: 18.4400 us (0.096042 us/matrix)
- n=8: 12.9080 us (0.067229 us/matrix)

### DType

- dtype=float16: 12.6630 us (0.065953 us/matrix)
- dtype=float32: 12.9080 us (0.067229 us/matrix)

### Mask

- mask_mode=absent: 12.9080 us (0.067229 us/matrix)
- mask_mode=full: 13.0240 us (0.067833 us/matrix)
- mask_mode=scalar: 12.8780 us (0.067073 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

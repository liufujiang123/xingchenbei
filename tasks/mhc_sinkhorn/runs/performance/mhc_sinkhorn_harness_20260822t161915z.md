# MhcSinkhorn local A3 performance — harness_20260822t161915z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 12.9820 | 12.982000 | 0.0770 | 12.7542, 12.8488, 12.9068, 12.9528, 12.9820, 12.9868, 13.0900, 13.4056, 13.6652 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 12.8588 | 3.214700 | 0.3111 | 12.6214, 12.7422, 12.7792, 12.8178, 12.8588, 12.9264, 12.9796, 13.0756, 13.1564 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.3008 | 0.277100 | 3.6088 | 12.9088, 13.1024, 13.1272, 13.2568, 13.3008, 13.3232, 13.3464, 13.4104, 14.0820 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 13.7420 | 0.071573 | 13.9718 | 13.3820, 13.5990, 13.6850, 13.7370, 13.7420, 13.9420, 14.0800, 14.1370, 14.2000 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 37.5530 | 0.048897 | 20.4511 | 37.4900, 37.4960, 37.5050, 37.5210, 37.5530, 37.5560, 37.5670, 37.9660, 37.9890 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.4276 | 0.069935 | 14.2989 | 13.0108, 13.1300, 13.1968, 13.2068, 13.4276, 13.4680, 13.5472, 13.5764, 13.6644 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.2016 | 0.068758 | 14.5437 | 12.9076, 13.0368, 13.0988, 13.1996, 13.2016, 13.3152, 13.3944, 13.4280, 13.4296 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 46.2160 | 0.240708 | 4.1544 | 44.9180, 45.0060, 45.0080, 45.9960, 46.2160, 46.2900, 46.2960, 46.3540, 46.3600 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.5680 | 0.101917 | 9.8119 | 19.2760, 19.2876, 19.3016, 19.5468, 19.5680, 19.5740, 19.5760, 19.5864, 19.5868 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 21.5420 | 0.112198 | 8.9128 | 21.4730, 21.4750, 21.5380, 21.5410, 21.5420, 21.5490, 21.5680, 21.9800, 21.9970 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 13.7990 | 0.071870 | 13.9141 | 13.4170, 13.5000, 13.7280, 13.7620, 13.7990, 13.8400, 14.0420, 14.3820, 14.4360 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 13.8330 | 0.072047 | 13.8799 | 13.5590, 13.5990, 13.6320, 13.7990, 13.8330, 13.8920, 14.0980, 14.1550, 14.3980 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 13.7880 | 0.071812 | 13.9252 | 13.5300, 13.6620, 13.7410, 13.7850, 13.7880, 13.8380, 13.8920, 13.9270, 14.0230 |

## Matrix-count amortization

- matrix_count=1: 12.9820 us (12.982000 us/matrix)
- matrix_count=4: 12.8588 us (3.214700 us/matrix)
- matrix_count=48: 13.3008 us (0.277100 us/matrix)
- matrix_count=192: 13.7420 us (0.071573 us/matrix)
- matrix_count=768: 37.5530 us (0.048897 us/matrix)

## Iteration cost model

- iterations=1: 13.4276 us (0.069935 us/matrix)
- iterations=5: 13.2016 us (0.068758 us/matrix)
- iterations=20: 13.7420 us (0.071573 us/matrix)
- iterations=100: 46.2160 us (0.240708 us/matrix)

Least-squares proxy: `T(iterations) = 10.6782 us + iterations * 0.3482 us`, R²=0.972541.

## Orthogonal comparisons

### N

- n=4: 19.5680 us (0.101917 us/matrix)
- n=6: 21.5420 us (0.112198 us/matrix)
- n=8: 13.7420 us (0.071573 us/matrix)

### DType

- dtype=float16: 13.7990 us (0.071870 us/matrix)
- dtype=float32: 13.7420 us (0.071573 us/matrix)

### Mask

- mask_mode=absent: 13.7420 us (0.071573 us/matrix)
- mask_mode=full: 13.7880 us (0.071812 us/matrix)
- mask_mode=scalar: 13.8330 us (0.072047 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

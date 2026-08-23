# MhcSinkhorn local A3 performance — harness_20260822t163510z

> Evidence class: `benchmark_observed`.
> Proxy metric: CANN 8.5 ACL runtime event time on local Ascend910_9382; this is not a CANNJudge 910B score.
> Warmup=5; active samples=9; each active sample averages repeated identical ACLNN launches.

| Case | Shape | Matrices | N | Iterations | DType | Mask | Median us | us/matrix | matrices/us | Samples us |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| mc_001 | [8, 8] | 1 | 8 | 20 | float32 | absent | 13.5408 | 13.540800 | 0.0739 | 13.2272, 13.3516, 13.4120, 13.5068, 13.5408, 13.8016, 13.8118, 13.8526, 14.7488 |
| mc_004 | [4, 8, 8] | 4 | 8 | 20 | float32 | absent | 13.2608 | 3.315200 | 0.3016 | 13.0420, 13.1138, 13.1764, 13.2416, 13.2608, 13.3882, 13.4182, 13.4626, 13.6724 |
| mc_048 | [48, 8, 8] | 48 | 8 | 20 | float32 | absent | 13.7244 | 0.285925 | 3.4974 | 13.3824, 13.4076, 13.5648, 13.6508, 13.7244, 13.7592, 13.8096, 13.8816, 14.4072 |
| mc_192 | [192, 8, 8] | 192 | 8 | 20 | float32 | absent | 14.8150 | 0.077161 | 12.9598 | 14.7270, 14.7330, 14.7350, 14.8090, 14.8150, 14.8410, 14.8770, 14.8780, 15.6050 |
| mc_768 | [768, 8, 8] | 768 | 8 | 20 | float32 | absent | 45.1270 | 0.058759 | 17.0186 | 44.4570, 44.9820, 45.1160, 45.1240, 45.1270, 45.1360, 45.1400, 45.1440, 45.4710 |
| iter_001 | [192, 8, 8] | 192 | 8 | 1 | float32 | absent | 13.5712 | 0.070683 | 14.1476 | 12.9504, 13.2072, 13.4576, 13.5320, 13.5712, 13.7604, 13.7872, 13.8264, 14.5920 |
| iter_005 | [192, 8, 8] | 192 | 8 | 5 | float32 | absent | 13.5832 | 0.070746 | 14.1351 | 13.1916, 13.2612, 13.4948, 13.5372, 13.5832, 13.5936, 13.6604, 13.7064, 14.4592 |
| iter_100 | [192, 8, 8] | 192 | 8 | 100 | float32 | absent | 53.8840 | 0.280646 | 3.5632 | 53.8540, 53.8620, 53.8680, 53.8760, 53.8840, 53.8880, 53.8940, 53.8940, 55.5040 |
| n_004 | [192, 4, 4] | 192 | 4 | 20 | float32 | absent | 19.3400 | 0.100729 | 9.9276 | 19.3100, 19.3372, 19.3376, 19.3392, 19.3400, 19.3452, 19.3468, 19.3544, 19.3640 |
| n_006 | [192, 6, 6] | 192 | 6 | 20 | float32 | absent | 22.1960 | 0.115604 | 8.6502 | 21.5820, 22.1530, 22.1800, 22.1910, 22.1960, 22.2010, 22.2060, 22.2110, 22.2520 |
| dtype_fp16 | [192, 8, 8] | 192 | 8 | 20 | float16 | absent | 15.4570 | 0.080505 | 12.4216 | 15.4310, 15.4320, 15.4360, 15.4370, 15.4570, 15.4640, 15.4640, 15.4670, 15.5040 |
| mask_scalar | [192, 8, 8] | 192 | 8 | 20 | float32 | scalar | 15.6520 | 0.081521 | 12.2668 | 15.5980, 15.6370, 15.6460, 15.6510, 15.6520, 15.6890, 15.7280, 15.8730, 15.9470 |
| mask_full | [192, 8, 8] | 192 | 8 | 20 | float32 | full | 15.6290 | 0.081401 | 12.2849 | 15.3130, 15.6080, 15.6100, 15.6210, 15.6290, 15.6490, 15.6590, 15.6680, 15.6830 |

## Matrix-count amortization

- matrix_count=1: 13.5408 us (13.540800 us/matrix)
- matrix_count=4: 13.2608 us (3.315200 us/matrix)
- matrix_count=48: 13.7244 us (0.285925 us/matrix)
- matrix_count=192: 14.8150 us (0.077161 us/matrix)
- matrix_count=768: 45.1270 us (0.058759 us/matrix)

## Iteration cost model

- iterations=1: 13.5712 us (0.070683 us/matrix)
- iterations=5: 13.5832 us (0.070746 us/matrix)
- iterations=20: 14.8150 us (0.077161 us/matrix)
- iterations=100: 53.8840 us (0.280646 us/matrix)

Least-squares proxy: `T(iterations) = 10.5633 us + iterations * 0.4254 us`, R²=0.978076.

## Orthogonal comparisons

### N

- n=4: 19.3400 us (0.100729 us/matrix)
- n=6: 22.1960 us (0.115604 us/matrix)
- n=8: 14.8150 us (0.077161 us/matrix)

### DType

- dtype=float16: 15.4570 us (0.080505 us/matrix)
- dtype=float32: 14.8150 us (0.077161 us/matrix)

### Mask

- mask_mode=absent: 14.8150 us (0.077161 us/matrix)
- mask_mode=full: 15.6290 us (0.081401 us/matrix)
- mask_mode=scalar: 15.6520 us (0.081521 us/matrix)

## Interpretation boundary

- Tensor allocation, input copies, and output copies are outside the timed event interval.
- Each launch intentionally follows the public ACLNN GetWorkspaceSize/executor path. Host enqueue gaps can appear as device-stream idle time between repeated launches.
- Use these numbers for same-machine, same-path candidate screening and workload-shape diagnosis only; platform ranking requires CANNJudge evidence.

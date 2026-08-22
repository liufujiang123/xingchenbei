# MhcExpand local A3 performance — candidate3_m1

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 14.6876 | 14.3082 | 15.7456 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 14.3474 | 14.1688 | 14.8046 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 14.6460 | 14.2653 | 15.0767 | 6.712 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5260 | 13.0487 | 14.2627 | 7.268 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8267 | 13.2180 | 14.0267 | 47.120 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.8593 | 13.1940 | 14.7393 | 47.009 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5640 | 12.0820 | 16.0940 | 2267.931 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.8000 | 13.1520 | 14.4820 | 2393.489 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.1680 | 14.4540 | 17.7420 | 2765.232 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.6420 | 14.5020 | 39.3400 | 2681.437 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3692 | 13.0506 | 13.8202 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2770 | 13.0084 | 13.8234 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.4347 | 12.8980 | 14.1620 | 7.317 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.3573 | 12.9593 | 14.6200 | 7.360 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3360 | 13.0947 | 14.6887 | 48.853 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7147 | 13.0987 | 14.1787 | 47.505 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.7100 | 13.1100 | 16.5440 | 305.930 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.8220 | 13.2280 | 14.7600 | 303.451 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 23.4820 | 23.1640 | 24.5620 | 1406.615 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 23.4900 | 23.3200 | 24.3760 | 1406.136 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.1980 | 28.6680 | 29.3280 | 1436.504 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 27.9860 | 27.7020 | 29.1120 | 1498.715 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

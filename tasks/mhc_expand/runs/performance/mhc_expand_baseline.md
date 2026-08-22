# MhcExpand local A3 performance — baseline

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2772 | 13.0930 | 13.9678 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0170 | 12.8600 | 13.5044 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.1387 | 12.7227 | 14.0207 | 7.482 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5380 | 13.2240 | 14.3767 | 7.261 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.9100 | 13.0147 | 14.3693 | 46.838 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6053 | 13.3667 | 14.9473 | 47.886 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3700 | 13.8440 | 15.4140 | 2298.549 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.9320 | 13.7080 | 14.8920 | 2370.811 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.3000 | 14.1940 | 15.5680 | 2933.080 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.4280 | 13.9160 | 16.1160 | 2718.631 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2044 | 12.8954 | 13.6674 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1338 | 13.0734 | 13.3542 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8827 | 13.2907 | 14.4267 | 7.081 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4580 | 13.1953 | 14.3293 | 7.305 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6120 | 13.1833 | 13.8527 | 47.863 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8420 | 13.3327 | 14.5360 | 47.068 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 25.5460 | 24.0700 | 26.1540 | 1292.967 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 25.4460 | 25.0520 | 25.5660 | 1298.049 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.4660 | 29.3060 | 30.4900 | 1423.439 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.5220 | 29.2700 | 29.7380 | 1420.738 |

## Summary

- Cases: 20
- Forward cases: 10
- Backward cases: 10
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

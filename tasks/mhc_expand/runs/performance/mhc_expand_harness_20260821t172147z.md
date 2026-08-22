# MhcExpand local A3 performance — harness_20260821t172147z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2976 | 13.1528 | 13.6968 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2982 | 12.9286 | 13.5450 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4047 | 12.6987 | 13.5933 | 7.334 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.7693 | 13.0213 | 14.2273 | 7.139 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3907 | 12.7200 | 13.7787 | 48.654 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3613 | 12.7920 | 13.6233 | 48.761 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.2380 | 13.6780 | 16.6900 | 2319.858 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.7880 | 13.2060 | 16.5440 | 2395.572 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.8520 | 13.6020 | 15.6720 | 3027.941 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.8840 | 13.3640 | 16.7460 | 2817.995 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2052 | 13.0534 | 13.4038 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.8240 | 12.7406 | 13.3426 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.0767 | 12.9173 | 13.4387 | 7.518 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.1173 | 13.0440 | 13.3920 | 7.494 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.1827 | 12.9860 | 13.6653 | 49.422 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.0000 | 12.8880 | 13.7973 | 50.116 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.4020 | 13.6520 | 14.5840 | 291.231 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3440 | 13.5800 | 15.1400 | 292.408 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 20.0040 | 19.9040 | 20.5420 | 1651.177 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 19.9880 | 19.9560 | 20.1100 | 1652.499 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 20.5560 | 20.4820 | 20.9300 | 2040.428 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 22.0320 | 20.6740 | 22.1240 | 1903.733 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

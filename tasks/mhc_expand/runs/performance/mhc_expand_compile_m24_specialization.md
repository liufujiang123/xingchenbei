# MhcExpand local A3 performance — compile_m24_specialization

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3108 | 13.1878 | 14.1728 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0762 | 13.0242 | 13.3588 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.3093 | 13.0460 | 13.4400 | 7.386 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.3333 | 13.0480 | 13.4067 | 7.373 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4353 | 13.1140 | 13.6140 | 48.492 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3947 | 12.8827 | 13.5867 | 48.640 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4000 | 13.9480 | 14.8260 | 2293.760 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.9880 | 13.9020 | 14.7240 | 2361.320 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2840 | 14.1260 | 15.0020 | 2936.365 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.1420 | 13.8700 | 14.8760 | 2965.849 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.0942 | 12.9378 | 13.2348 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0434 | 12.9826 | 13.4380 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.1953 | 12.9860 | 13.5247 | 7.450 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2080 | 12.8233 | 13.5327 | 7.443 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.1447 | 13.0993 | 13.3153 | 49.565 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.3567 | 13.1493 | 13.5393 | 48.778 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.8480 | 13.0300 | 14.6680 | 302.882 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.4180 | 12.9740 | 14.0180 | 312.588 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.5100 | 13.8140 | 15.1240 | 2276.371 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.4500 | 14.3900 | 15.1060 | 2285.823 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.3500 | 13.9320 | 15.1260 | 2922.860 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.0560 | 12.7020 | 14.6080 | 2983.995 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

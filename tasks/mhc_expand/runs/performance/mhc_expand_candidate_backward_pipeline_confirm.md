# MhcExpand local A3 performance — candidate_backward_pipeline_confirm

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 15.7840 | 15.3044 | 16.8494 | 0.006 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 15.2898 | 15.0392 | 15.5878 | 0.006 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 15.5367 | 15.4287 | 16.2727 | 6.327 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 15.9047 | 15.4407 | 16.5580 | 6.181 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 15.9320 | 15.0840 | 17.2740 | 40.893 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.9047 | 13.2240 | 14.5087 | 46.855 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5920 | 14.2840 | 16.9260 | 2263.579 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.7100 | 14.5020 | 16.4020 | 2245.421 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.5400 | 14.2100 | 14.8960 | 2884.666 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.2960 | 14.1380 | 15.7400 | 2933.900 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2410 | 13.0296 | 13.6118 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1632 | 13.0288 | 13.4570 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8887 | 13.2927 | 14.4347 | 7.078 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7760 | 13.5027 | 14.2300 | 7.136 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.8280 | 13.4993 | 13.8833 | 47.115 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7600 | 13.4100 | 14.1827 | 47.348 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.3420 | 13.8720 | 15.4500 | 292.449 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.9120 | 14.6020 | 28.4240 | 281.270 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.7940 | 13.8640 | 29.6240 | 2232.672 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.7960 | 14.6100 | 17.1800 | 2232.370 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.4840 | 13.6100 | 15.9380 | 2895.819 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.6180 | 14.2960 | 15.7520 | 2869.273 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

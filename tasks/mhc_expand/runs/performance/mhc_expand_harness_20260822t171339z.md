# MhcExpand local A3 performance — harness_20260822t171339z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.0846 | 12.7022 | 13.5902 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.8244 | 12.6836 | 14.3196 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9300 | 12.8167 | 13.1807 | 7.603 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.9833 | 12.7467 | 13.3813 | 7.572 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 12.9180 | 12.4553 | 13.1053 | 50.434 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.7320 | 12.5980 | 12.9927 | 51.171 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.9860 | 11.9060 | 14.9620 | 2361.658 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 12.8260 | 12.7460 | 13.7660 | 2575.249 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.2960 | 12.9160 | 14.1700 | 3154.561 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 13.7480 | 13.1000 | 14.2360 | 3050.847 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.6012 | 12.4872 | 12.8076 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5298 | 12.4396 | 12.7122 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.6000 | 12.4247 | 13.2320 | 7.802 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.5847 | 12.3473 | 12.8167 | 7.811 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.6533 | 12.5360 | 13.3647 | 51.489 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.0973 | 12.8127 | 13.1640 | 49.744 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1960 | 13.6040 | 15.5040 | 295.457 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.8640 | 13.1860 | 14.1360 | 302.532 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.8520 | 13.6120 | 14.5040 | 2384.504 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.8540 | 13.5560 | 14.4680 | 2384.159 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 13.4840 | 13.2460 | 13.9480 | 3110.579 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 17.6880 | 15.4420 | 18.1760 | 2371.271 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

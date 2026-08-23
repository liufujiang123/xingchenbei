# MhcExpand local A3 performance — harness_20260822t172727z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.8566 | 12.7354 | 13.3636 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.6020 | 10.8362 | 12.8362 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.8453 | 12.1667 | 13.1080 | 7.653 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.3613 | 12.0647 | 12.4987 | 7.953 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 12.2520 | 11.9553 | 12.4080 | 53.176 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.0800 | 11.9687 | 12.2753 | 53.933 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 12.9960 | 12.6060 | 13.4360 | 2541.562 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 12.8040 | 12.1920 | 13.1960 | 2579.674 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.9200 | 13.5880 | 14.1220 | 3013.149 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.3140 | 13.7800 | 14.7040 | 2930.211 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.3118 | 12.1652 | 12.7520 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.4260 | 12.1660 | 12.6636 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.8180 | 12.4473 | 13.0213 | 7.669 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.5920 | 12.4567 | 13.0273 | 7.807 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.7467 | 12.4980 | 13.1273 | 51.112 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 12.8233 | 12.7020 | 13.1207 | 50.807 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 12.6080 | 12.3940 | 12.8920 | 332.670 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 12.6280 | 12.4440 | 13.1300 | 332.143 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.4300 | 13.1340 | 14.6900 | 2459.430 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.5200 | 14.3900 | 16.0740 | 2274.803 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 13.3320 | 12.8760 | 13.9860 | 3146.043 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 13.5020 | 12.5500 | 14.2240 | 3106.432 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

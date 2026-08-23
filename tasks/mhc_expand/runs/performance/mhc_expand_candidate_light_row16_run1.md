# MhcExpand local A3 performance — candidate_light_row16_run1

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.9360 | 12.7782 | 13.2672 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.7688 | 12.6174 | 13.0408 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9900 | 12.8107 | 13.5640 | 7.568 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.9380 | 12.8447 | 13.1280 | 7.598 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.0420 | 12.8127 | 13.8367 | 49.955 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3427 | 12.6847 | 17.6133 | 48.829 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.2760 | 13.9640 | 15.0720 | 2313.683 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.6060 | 12.2400 | 15.8380 | 2427.616 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.8900 | 13.4900 | 14.1300 | 3019.657 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 13.6960 | 13.5340 | 14.5220 | 3062.430 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.3526 | 12.1576 | 12.4692 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5210 | 12.4442 | 12.7464 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.8220 | 12.6080 | 13.2827 | 7.667 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.8187 | 12.6880 | 12.9380 | 7.669 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.7640 | 12.6300 | 13.1847 | 51.043 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 12.8747 | 12.6940 | 13.5187 | 50.604 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.5640 | 12.9700 | 14.3260 | 309.223 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.7860 | 13.4280 | 14.4220 | 304.244 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.8580 | 13.5760 | 14.2200 | 2383.471 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.9140 | 13.6620 | 14.4760 | 2373.878 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.3080 | 13.9040 | 14.6860 | 2931.440 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 13.2960 | 12.6300 | 14.5980 | 3154.561 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — harness_20260822t155400z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3070 | 13.0396 | 13.8898 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1356 | 12.8692 | 13.2740 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2640 | 12.8300 | 13.8047 | 7.411 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.2367 | 12.9053 | 14.3253 | 7.427 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.7127 | 12.9007 | 13.9540 | 47.512 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.4447 | 12.8473 | 13.9760 | 48.459 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3580 | 13.7640 | 15.8840 | 2300.470 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1960 | 13.8540 | 14.5900 | 2326.722 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.8240 | 14.6500 | 15.4020 | 2829.401 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.8760 | 14.7600 | 15.8780 | 2819.511 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1862 | 13.0340 | 13.7658 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0846 | 12.9926 | 13.4338 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.1387 | 12.7040 | 13.9693 | 7.482 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.8353 | 12.5940 | 13.5253 | 7.659 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3233 | 12.9860 | 13.7987 | 48.900 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8147 | 13.0093 | 14.1620 | 47.161 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1620 | 13.8580 | 15.6580 | 296.166 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.5240 | 13.7080 | 15.6540 | 288.784 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.6480 | 14.3080 | 16.5040 | 2254.925 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.6900 | 14.3180 | 16.8960 | 2248.478 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.0080 | 13.6000 | 16.2240 | 2794.712 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.3560 | 13.3140 | 15.6220 | 2921.638 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

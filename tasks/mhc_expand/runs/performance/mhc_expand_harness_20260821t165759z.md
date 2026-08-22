# MhcExpand local A3 performance — harness_20260821t165759z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2382 | 12.9152 | 13.7328 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0050 | 12.7896 | 13.4560 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5613 | 13.1187 | 13.8840 | 7.249 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.3533 | 12.9033 | 13.9427 | 7.362 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4713 | 12.9560 | 13.9580 | 48.363 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.1933 | 12.8547 | 13.6747 | 49.382 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 15.0340 | 14.2120 | 15.9180 | 2197.030 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.7940 | 13.3800 | 14.7700 | 2394.530 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.9200 | 14.8780 | 15.2900 | 2811.196 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.2160 | 15.0920 | 16.1560 | 2756.509 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.0934 | 12.7540 | 13.3018 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0120 | 12.8874 | 13.5960 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6040 | 12.6013 | 13.8720 | 7.226 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4733 | 13.0680 | 13.6560 | 7.296 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.9793 | 13.2260 | 14.5973 | 46.605 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8100 | 13.2493 | 14.3780 | 47.177 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.8560 | 13.4920 | 15.6500 | 302.707 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.7640 | 13.0300 | 17.5820 | 304.730 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 22.8060 | 22.7260 | 24.0340 | 1448.309 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 22.9840 | 22.9100 | 23.8700 | 1437.093 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 27.4440 | 27.2780 | 29.0500 | 1528.314 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 27.3580 | 27.2660 | 27.5420 | 1533.118 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — harness_20260821t171630z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3668 | 13.2286 | 13.6464 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3250 | 13.1944 | 13.5528 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5353 | 13.0640 | 13.7140 | 7.263 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.7193 | 13.1967 | 14.1673 | 7.165 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.9233 | 13.1313 | 14.1273 | 46.793 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.7680 | 13.3467 | 14.0240 | 47.321 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3280 | 14.1980 | 17.5940 | 2305.286 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.4440 | 13.9660 | 17.8260 | 2286.773 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.9420 | 14.6060 | 15.1760 | 2807.057 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.9220 | 14.4480 | 19.6620 | 2810.819 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2762 | 13.1858 | 13.5464 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2942 | 13.1864 | 13.3818 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.7260 | 13.1727 | 14.0227 | 7.162 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5387 | 13.2267 | 13.7333 | 7.261 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.8540 | 13.3700 | 14.2380 | 47.027 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.0113 | 13.2760 | 14.3647 | 46.499 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.3800 | 14.0320 | 15.5280 | 291.676 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.7160 | 14.1280 | 15.9200 | 285.017 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 20.8560 | 20.7340 | 40.5260 | 1583.724 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 20.9740 | 20.9020 | 21.5180 | 1574.814 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 21.3460 | 21.2500 | 21.9800 | 1964.913 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 21.7100 | 21.6500 | 21.7560 | 1931.969 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — aligned_datacopy_full

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.8218 | 12.7338 | 13.3290 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.8368 | 12.6256 | 13.1624 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.3047 | 12.8993 | 13.7540 | 7.389 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.6047 | 13.3853 | 13.9067 | 7.226 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.9847 | 13.3020 | 21.3873 | 46.587 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 14.0400 | 13.6300 | 14.2620 | 46.404 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5600 | 13.8380 | 15.3060 | 2268.554 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.7680 | 14.3980 | 15.7820 | 2236.602 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.8400 | 13.9080 | 15.8500 | 2826.351 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5180 | 13.5720 | 15.0920 | 2889.037 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9032 | 12.6444 | 13.1268 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.7600 | 12.6836 | 13.0514 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.0773 | 5.0327 | 13.4400 | 7.517 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2560 | 13.0933 | 14.4767 | 7.416 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4127 | 13.0740 | 13.8887 | 48.574 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.5907 | 13.0860 | 13.9427 | 47.938 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.9680 | 13.5660 | 15.5640 | 300.279 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.9740 | 13.9040 | 14.8420 | 300.151 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.3700 | 13.8160 | 14.7480 | 2298.549 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.9280 | 13.4680 | 15.2860 | 2212.630 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.0440 | 13.4880 | 14.7560 | 2986.545 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.0880 | 13.4620 | 15.8580 | 2977.217 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

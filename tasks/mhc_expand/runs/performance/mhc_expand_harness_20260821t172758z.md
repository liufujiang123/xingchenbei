# MhcExpand local A3 performance — harness_20260821t172758z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.0584 | 12.9366 | 13.7562 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9208 | 12.7610 | 13.2280 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.1727 | 13.0347 | 13.4653 | 7.463 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4447 | 13.0800 | 13.7067 | 7.312 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.2720 | 13.0440 | 14.0273 | 49.089 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.2687 | 13.1393 | 13.7013 | 49.101 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.0020 | 13.2520 | 14.8780 | 2358.959 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1940 | 13.8160 | 15.2120 | 2327.050 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.5420 | 14.3300 | 15.5220 | 2884.269 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.1920 | 13.8180 | 14.4520 | 2955.400 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9072 | 12.6478 | 13.1372 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.8004 | 12.7536 | 12.9894 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.2853 | 12.9140 | 13.4020 | 7.399 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.1940 | 13.0827 | 13.7473 | 7.451 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.2180 | 12.9533 | 13.7267 | 49.290 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.2433 | 13.1627 | 13.6167 | 49.195 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.8340 | 13.6080 | 14.3580 | 303.188 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.1840 | 13.7640 | 15.0520 | 295.707 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.3380 | 14.0860 | 14.9400 | 2303.679 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.7380 | 14.5220 | 15.7320 | 2241.155 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.5820 | 14.4140 | 15.4240 | 2876.357 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.6240 | 14.0780 | 15.8220 | 2868.096 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

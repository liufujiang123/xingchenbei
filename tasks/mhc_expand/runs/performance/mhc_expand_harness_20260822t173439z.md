# MhcExpand local A3 performance — harness_20260822t173439z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3048 | 13.0844 | 13.6120 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.8898 | 12.7372 | 13.1462 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.8373 | 12.3600 | 14.1720 | 7.658 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1000 | 12.3580 | 13.6947 | 7.504 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 12.9000 | 12.3327 | 13.6400 | 50.505 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.9953 | 12.5347 | 14.5273 | 50.134 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.9240 | 13.7500 | 15.3800 | 2372.174 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.3180 | 13.8860 | 14.4580 | 2306.897 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1380 | 13.3860 | 14.6780 | 2966.688 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.2700 | 14.1840 | 15.3400 | 2939.246 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2768 | 12.9618 | 13.4196 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0788 | 12.9926 | 13.3502 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.2800 | 12.7487 | 14.1907 | 7.402 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.8833 | 12.7400 | 14.5460 | 7.630 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.9780 | 12.8520 | 13.5727 | 50.201 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7400 | 12.9287 | 13.9580 | 47.417 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1660 | 13.3480 | 14.7640 | 296.082 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.3420 | 12.9960 | 14.5180 | 314.368 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.8640 | 14.2640 | 16.3600 | 2222.157 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.0380 | 14.4520 | 16.0460 | 2196.445 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 13.5140 | 13.2420 | 15.1860 | 3103.673 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.5160 | 13.5340 | 14.7380 | 2889.435 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

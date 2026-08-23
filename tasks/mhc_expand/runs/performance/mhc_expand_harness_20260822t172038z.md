# MhcExpand local A3 performance — harness_20260822t172038z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.4932 | 13.3146 | 13.7656 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1918 | 12.9726 | 13.4554 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0407 | 12.7407 | 13.5913 | 7.538 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.3673 | 12.6527 | 14.0400 | 7.354 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.7807 | 12.8887 | 14.2573 | 47.277 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6693 | 13.1433 | 14.9453 | 47.662 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.6920 | 14.3940 | 15.8560 | 2248.172 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.5360 | 14.1120 | 15.3660 | 2272.300 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.0380 | 13.9640 | 15.4420 | 2789.137 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.3220 | 15.0880 | 15.9300 | 2737.439 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3236 | 12.9912 | 13.4766 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1644 | 12.9944 | 13.8204 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.3527 | 12.6960 | 14.1540 | 7.362 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5887 | 12.7853 | 14.1753 | 7.234 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4787 | 13.1740 | 13.6807 | 48.336 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6380 | 13.0733 | 18.2233 | 47.772 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.6560 | 13.3300 | 14.5220 | 307.140 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.8900 | 12.9180 | 15.0540 | 301.966 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.8440 | 13.1000 | 21.2660 | 2225.151 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.5520 | 14.2240 | 17.2340 | 2269.801 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.8060 | 14.3840 | 16.8900 | 2832.841 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.9380 | 13.8220 | 15.4460 | 2807.808 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

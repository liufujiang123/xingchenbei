# MhcExpand local A3 performance — forward_row_batch_repeat2

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1282 | 12.8500 | 13.6406 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.8446 | 12.6960 | 13.2100 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2813 | 12.9707 | 13.4860 | 7.402 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5093 | 13.0740 | 14.4913 | 7.277 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.1280 | 12.7313 | 14.3633 | 49.628 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3607 | 12.8300 | 13.5033 | 48.763 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4980 | 13.9180 | 16.1720 | 2278.255 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.4480 | 12.9980 | 15.4940 | 2456.138 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.8700 | 13.2420 | 18.5240 | 3024.012 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 13.6800 | 13.4380 | 14.7100 | 3066.012 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9910 | 12.7704 | 13.3156 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9624 | 12.8440 | 13.2374 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.5213 | 13.0620 | 13.8660 | 7.270 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4727 | 13.1173 | 13.5647 | 7.297 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.7547 | 13.2113 | 14.0653 | 47.366 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7220 | 13.3147 | 13.9360 | 47.479 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.9860 | 13.7300 | 15.6060 | 299.893 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.9080 | 13.8000 | 15.8020 | 301.575 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 23.8840 | 22.9280 | 24.7700 | 1382.940 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 24.0220 | 23.9740 | 24.1420 | 1374.996 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 28.9900 | 28.8400 | 29.1100 | 1446.811 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.0320 | 28.9080 | 29.0620 | 1444.718 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

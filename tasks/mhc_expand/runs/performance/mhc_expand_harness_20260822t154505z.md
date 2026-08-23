# MhcExpand local A3 performance — harness_20260822t154505z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2858 | 12.9532 | 13.8940 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0598 | 12.7094 | 13.2070 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2773 | 12.7633 | 14.0993 | 7.404 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5253 | 12.6967 | 13.8513 | 7.268 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4687 | 12.6647 | 14.0840 | 48.372 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.1387 | 12.8440 | 14.1913 | 49.587 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4580 | 13.9200 | 16.1660 | 2284.558 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.7660 | 12.0140 | 15.3080 | 2236.905 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.8960 | 13.0720 | 14.5200 | 3018.353 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.7300 | 14.3020 | 16.0180 | 2847.457 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.0660 | 12.8280 | 13.6234 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1694 | 12.8210 | 13.3296 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.4167 | 12.9247 | 14.1960 | 7.327 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.1160 | 12.8193 | 14.1827 | 7.495 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.0873 | 12.7493 | 13.5040 | 49.782 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.4647 | 12.8720 | 13.7413 | 48.387 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.5960 | 13.1760 | 14.8220 | 308.495 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.1080 | 12.9640 | 14.3480 | 319.980 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.4700 | 13.3700 | 15.3800 | 2282.664 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.3140 | 14.0380 | 15.5920 | 2307.541 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.3700 | 12.7720 | 15.9200 | 2918.792 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.0700 | 12.7840 | 15.6780 | 2981.026 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

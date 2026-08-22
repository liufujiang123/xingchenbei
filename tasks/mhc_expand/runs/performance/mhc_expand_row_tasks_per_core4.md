# MhcExpand local A3 performance — row_tasks_per_core4

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1320 | 12.9940 | 13.8962 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1602 | 13.0140 | 13.3564 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.3380 | 13.0960 | 13.8427 | 7.370 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4333 | 13.2133 | 13.8093 | 7.318 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3067 | 13.1193 | 14.0693 | 48.961 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.1480 | 12.8607 | 13.3047 | 49.552 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3000 | 13.8220 | 14.6980 | 2309.800 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.0340 | 13.8200 | 14.6960 | 2353.580 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.3440 | 14.2080 | 14.7700 | 2924.083 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.3160 | 14.1720 | 14.7480 | 2929.802 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1348 | 13.0566 | 13.3102 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.2078 | 12.1134 | 13.2574 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.0833 | 12.9680 | 13.3140 | 7.514 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.5053 | 12.4133 | 13.4167 | 7.861 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.5400 | 12.4153 | 12.9527 | 51.955 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 12.8933 | 12.6160 | 13.1407 | 50.531 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.4000 | 13.0320 | 14.6900 | 313.008 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.1020 | 12.9400 | 13.4120 | 320.127 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.9120 | 13.8500 | 14.5540 | 2374.220 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.6660 | 14.0960 | 15.1980 | 2252.158 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.6680 | 14.3520 | 15.2000 | 2859.493 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.1780 | 14.0620 | 14.5460 | 2958.318 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

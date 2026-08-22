# MhcExpand local A3 performance — forward_row_batch_repeat1

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.6430 | 11.1762 | 13.6702 | 0.008 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.7750 | 12.6572 | 12.8970 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9640 | 12.7593 | 13.7807 | 7.583 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4513 | 12.9513 | 13.7760 | 7.308 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3887 | 12.9020 | 13.6260 | 48.661 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.7613 | 12.3753 | 13.5067 | 51.053 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.0660 | 12.7080 | 13.6080 | 2527.946 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 12.7580 | 12.5620 | 13.5620 | 2588.975 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2320 | 13.8640 | 14.3980 | 2947.094 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 12.9680 | 12.7200 | 13.2020 | 3234.349 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.5740 | 12.4106 | 12.7206 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5322 | 12.4674 | 12.8904 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.7820 | 12.6887 | 13.5940 | 7.691 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.9000 | 12.7827 | 13.0353 | 7.620 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.0740 | 12.6740 | 13.9360 | 49.832 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.0427 | 12.7547 | 13.8720 | 49.952 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.1480 | 12.8240 | 13.4980 | 319.007 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.7080 | 13.3020 | 17.3700 | 305.975 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 23.0600 | 22.9940 | 23.1240 | 1432.357 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 23.1200 | 23.0040 | 23.2460 | 1428.639 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.0420 | 27.5880 | 29.1100 | 1444.220 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 27.6060 | 27.4760 | 27.7200 | 1519.345 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

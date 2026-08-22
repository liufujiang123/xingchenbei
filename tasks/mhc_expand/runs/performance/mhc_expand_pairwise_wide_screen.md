# MhcExpand local A3 performance — pairwise_wide_screen

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1692 | 12.9706 | 13.6914 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9068 | 12.7354 | 13.3806 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0813 | 12.8340 | 13.6873 | 7.515 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1467 | 12.9887 | 13.3587 | 7.477 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.2853 | 13.0740 | 13.8173 | 49.040 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.1027 | 12.8127 | 13.5453 | 49.723 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.2600 | 12.7400 | 13.9160 | 2490.961 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.0860 | 13.7800 | 15.2440 | 2344.892 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2520 | 13.8360 | 14.9800 | 2942.958 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 13.7220 | 13.3020 | 13.8940 | 3056.627 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.7940 | 12.7316 | 13.0448 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.6572 | 12.6136 | 13.1698 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.8820 | 12.7933 | 13.1413 | 7.631 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.9953 | 12.8607 | 13.2233 | 7.565 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.0547 | 12.7427 | 13.1307 | 49.906 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.0013 | 12.9127 | 13.7620 | 50.111 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.7140 | 13.6420 | 13.9000 | 305.841 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.7520 | 13.6680 | 14.3380 | 304.996 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 16.1720 | 16.0180 | 16.6980 | 2042.428 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 16.5420 | 15.5140 | 18.7620 | 1996.744 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.2440 | 13.8380 | 16.4280 | 2944.611 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.6340 | 13.9000 | 17.4060 | 2866.136 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

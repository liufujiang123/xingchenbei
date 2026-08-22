# MhcExpand local A3 performance — final

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2288 | 13.0668 | 13.7702 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9608 | 12.8610 | 13.3294 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2467 | 12.7340 | 13.5987 | 7.421 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4740 | 12.8313 | 14.1160 | 7.296 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4227 | 12.9347 | 14.1407 | 48.538 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.4387 | 13.0447 | 14.0880 | 48.480 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4300 | 14.0960 | 16.0780 | 2288.991 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6900 | 14.2860 | 15.0000 | 2248.478 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.1920 | 15.0860 | 15.5180 | 2760.864 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.6240 | 13.7160 | 18.5780 | 2684.526 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2848 | 13.1912 | 14.9116 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2154 | 13.1424 | 13.3542 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8487 | 13.1780 | 14.2760 | 7.098 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7173 | 13.5453 | 14.0727 | 7.166 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6973 | 13.4140 | 13.9427 | 47.565 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.0187 | 13.4720 | 14.6953 | 46.474 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.9920 | 13.7660 | 15.6140 | 299.764 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.2820 | 14.1500 | 15.7700 | 293.678 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 23.7300 | 23.6400 | 24.5040 | 1391.915 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 23.8260 | 23.6620 | 24.4980 | 1386.307 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 28.2900 | 28.1020 | 29.0360 | 1482.610 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 28.7480 | 28.6980 | 29.2820 | 1458.990 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

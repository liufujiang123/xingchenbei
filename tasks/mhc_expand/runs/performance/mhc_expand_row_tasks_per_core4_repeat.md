# MhcExpand local A3 performance — row_tasks_per_core4_repeat

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.8958 | 12.7826 | 13.5452 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0614 | 12.5596 | 14.3336 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0427 | 12.7740 | 13.4107 | 7.537 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.0713 | 12.8100 | 13.2233 | 7.521 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.0947 | 12.7600 | 16.8027 | 49.754 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.0367 | 12.6687 | 13.1180 | 49.975 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.7580 | 13.5740 | 14.3520 | 2400.795 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.5620 | 13.0940 | 14.1860 | 2435.492 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1400 | 13.7240 | 14.3080 | 2966.269 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4380 | 14.1040 | 14.6200 | 2905.045 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.7306 | 12.5408 | 12.9642 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5528 | 12.4592 | 12.7636 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.9160 | 12.6073 | 13.3580 | 7.611 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.9533 | 12.7587 | 13.7107 | 7.589 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.8147 | 12.9480 | 16.7320 | 47.161 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.1160 | 12.8247 | 13.3907 | 49.673 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.9660 | 13.7220 | 15.8660 | 300.322 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.0380 | 13.5340 | 14.3420 | 298.782 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.8680 | 13.6560 | 14.7500 | 2381.753 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.2620 | 14.1720 | 14.7080 | 2315.955 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 13.7120 | 13.3960 | 18.4820 | 3058.857 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.1360 | 13.8980 | 14.6740 | 2967.108 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

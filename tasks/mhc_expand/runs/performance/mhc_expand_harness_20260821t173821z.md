# MhcExpand local A3 performance — harness_20260821t173821z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2230 | 13.0372 | 13.7420 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2302 | 13.0462 | 13.4388 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4187 | 13.0600 | 13.5887 | 7.326 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5027 | 13.2307 | 14.0347 | 7.280 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.7413 | 13.2720 | 13.9120 | 47.412 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6033 | 13.2640 | 14.0047 | 47.893 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.1840 | 13.2520 | 14.8880 | 2328.690 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.2720 | 13.5000 | 16.1380 | 2314.332 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1240 | 13.8260 | 15.0880 | 2969.629 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.3640 | 13.8880 | 15.0820 | 2920.011 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2822 | 13.0370 | 13.4682 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2794 | 13.1198 | 13.6758 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6793 | 13.3127 | 13.9873 | 7.186 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.6767 | 13.3607 | 14.1040 | 7.188 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6060 | 13.3180 | 14.1753 | 47.884 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6627 | 13.0887 | 13.8733 | 47.685 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1200 | 13.2780 | 15.3160 | 297.047 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.6780 | 13.0740 | 15.8340 | 306.646 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 15.2740 | 13.3460 | 16.5000 | 2162.508 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.3020 | 14.1460 | 14.8120 | 2309.477 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.4120 | 13.3760 | 15.7300 | 2910.286 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.3800 | 13.5980 | 16.7020 | 2916.762 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

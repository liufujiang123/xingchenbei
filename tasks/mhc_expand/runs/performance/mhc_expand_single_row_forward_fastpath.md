# MhcExpand local A3 performance — single_row_forward_fastpath

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.5164 | 13.2270 | 14.1824 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.4784 | 13.2246 | 13.7538 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5740 | 12.9420 | 14.0173 | 7.242 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5613 | 13.2660 | 14.0000 | 7.249 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 14.1727 | 13.2300 | 19.6993 | 45.969 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.9067 | 13.3793 | 14.4967 | 46.849 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.6500 | 13.9020 | 16.3520 | 2254.617 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6820 | 14.5520 | 15.0080 | 2249.703 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.9160 | 14.5280 | 15.2060 | 2811.950 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.9100 | 14.7020 | 16.3860 | 2813.081 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.5234 | 13.2536 | 14.0136 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.3706 | 13.2690 | 13.7520 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.9087 | 13.3207 | 14.1913 | 7.068 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5807 | 13.2713 | 14.2733 | 7.239 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.9367 | 13.4607 | 14.7007 | 46.748 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.1440 | 13.3660 | 14.7927 | 46.063 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.3960 | 14.0540 | 15.7580 | 291.352 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.4100 | 14.0840 | 15.6420 | 291.069 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.5980 | 14.4480 | 16.8680 | 2262.649 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.7160 | 14.4760 | 15.9480 | 2244.506 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.5660 | 14.3920 | 16.0960 | 2879.517 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 15.4700 | 14.9040 | 19.8120 | 2711.250 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

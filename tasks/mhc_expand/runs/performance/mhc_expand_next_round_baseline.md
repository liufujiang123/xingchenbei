# MhcExpand local A3 performance — next_round_baseline

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.4612 | 13.2534 | 14.6314 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.4100 | 13.0646 | 13.5338 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.8593 | 13.0727 | 14.0867 | 7.093 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.9887 | 12.9340 | 14.2307 | 7.027 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.9513 | 13.1113 | 14.3633 | 46.699 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.7127 | 13.0960 | 13.8273 | 47.512 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5840 | 14.1660 | 15.7220 | 2264.821 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.2200 | 13.8600 | 15.9080 | 2322.795 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.0660 | 13.6780 | 16.2320 | 2981.874 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.2340 | 13.7220 | 17.3840 | 2946.680 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3824 | 13.1672 | 13.8220 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2900 | 13.0830 | 13.5994 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8253 | 13.0340 | 14.2740 | 7.110 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.8520 | 13.4153 | 14.1400 | 7.097 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.5787 | 12.9427 | 13.6993 | 47.980 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.0433 | 12.9873 | 14.6247 | 46.393 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.6100 | 13.9700 | 16.8880 | 287.084 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3980 | 14.0860 | 16.4880 | 291.312 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.6200 | 14.3020 | 15.8100 | 2259.244 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.6280 | 14.5360 | 17.2220 | 2258.008 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.3680 | 13.5960 | 16.8680 | 2919.198 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.6220 | 14.0820 | 22.9560 | 2868.489 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

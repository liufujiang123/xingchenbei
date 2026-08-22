# MhcExpand local A3 performance — compile_m_specialization

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3818 | 13.2534 | 14.0026 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2598 | 13.1058 | 13.4538 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4520 | 12.9640 | 13.6480 | 7.308 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.3520 | 12.9387 | 14.1040 | 7.362 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8440 | 13.1713 | 14.0160 | 47.061 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.5700 | 13.3973 | 14.4340 | 48.011 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4520 | 14.2280 | 17.4040 | 2285.507 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.4920 | 13.1080 | 14.5860 | 2279.198 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.6580 | 14.5560 | 15.3900 | 2861.444 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.7160 | 14.6020 | 15.6820 | 2850.166 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3750 | 13.1666 | 13.7332 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.4542 | 13.2344 | 16.3684 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8547 | 13.3707 | 14.8100 | 7.095 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5960 | 13.1333 | 13.8107 | 7.230 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6833 | 13.3147 | 15.6073 | 47.613 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8387 | 13.3120 | 14.3313 | 47.079 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.3860 | 13.9380 | 15.4400 | 291.555 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.4200 | 13.9240 | 15.8240 | 290.867 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.9640 | 14.5920 | 16.5360 | 2207.307 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.1480 | 14.7740 | 16.3620 | 2180.495 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.6440 | 14.4260 | 16.0780 | 2864.179 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.7200 | 14.2720 | 16.3900 | 2849.391 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

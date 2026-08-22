# MhcExpand local A3 performance — harness_20260821t170543z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1832 | 12.3328 | 14.0736 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1616 | 12.8768 | 13.4332 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4733 | 13.0413 | 13.9513 | 7.296 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.0200 | 12.8687 | 13.9513 | 7.550 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.5007 | 12.8787 | 13.8140 | 48.258 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.2167 | 12.7433 | 13.7287 | 49.295 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.9840 | 13.0980 | 15.3580 | 2361.995 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6540 | 14.4540 | 16.0960 | 2254.002 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.6360 | 13.0440 | 14.4680 | 3075.905 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5460 | 14.2020 | 16.0700 | 2883.476 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1846 | 13.0036 | 13.2680 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1928 | 13.0634 | 13.4484 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6553 | 12.8953 | 14.1433 | 7.199 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7273 | 13.1540 | 13.9707 | 7.161 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.8707 | 13.1647 | 14.5260 | 46.970 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6880 | 13.0087 | 14.2327 | 47.597 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.6680 | 12.9920 | 15.2060 | 306.870 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.6420 | 13.3860 | 15.7100 | 286.457 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 24.7840 | 24.6660 | 25.6460 | 1332.720 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 23.5060 | 23.2960 | 24.3920 | 1405.179 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.2420 | 28.8460 | 29.5940 | 1434.342 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.1480 | 29.0400 | 29.4900 | 1438.968 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

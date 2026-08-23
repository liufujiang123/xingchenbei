# MhcExpand local A3 performance — candidate_compile_init

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2242 | 12.8314 | 14.1466 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9798 | 12.7030 | 13.3022 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2493 | 12.6473 | 13.6953 | 7.420 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1173 | 12.8493 | 13.7087 | 7.494 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3420 | 12.6413 | 13.7707 | 48.832 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.0413 | 12.9060 | 13.9787 | 49.957 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.8380 | 13.5980 | 15.1100 | 2386.916 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.6920 | 13.3220 | 14.0940 | 2412.368 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1360 | 13.5300 | 14.2460 | 2967.108 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4360 | 14.2780 | 16.0780 | 2905.447 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1688 | 13.0680 | 13.3976 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9630 | 12.7514 | 13.2494 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6347 | 12.6760 | 16.6700 | 7.210 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4153 | 12.9973 | 13.6700 | 7.328 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3040 | 12.9333 | 13.6860 | 48.971 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6280 | 12.8040 | 13.8673 | 47.807 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.2500 | 12.8440 | 15.3820 | 316.551 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.0960 | 13.0020 | 16.6660 | 297.553 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.9620 | 13.7300 | 15.2420 | 2365.717 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.8380 | 13.1980 | 15.6600 | 2386.916 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.6340 | 14.2180 | 16.3200 | 2866.136 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.0640 | 13.4740 | 15.8580 | 2982.298 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

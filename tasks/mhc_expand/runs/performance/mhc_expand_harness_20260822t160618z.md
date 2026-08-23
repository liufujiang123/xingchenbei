# MhcExpand local A3 performance — harness_20260822t160618z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.8614 | 13.6530 | 14.3742 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.7040 | 13.4162 | 13.9662 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.8360 | 13.2073 | 14.0780 | 7.105 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.8200 | 13.3747 | 14.6180 | 7.113 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8200 | 13.3380 | 14.5693 | 47.143 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.9073 | 13.5760 | 14.1987 | 46.847 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.6480 | 14.1860 | 17.0640 | 2254.925 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6320 | 13.9840 | 15.2680 | 2257.391 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.5320 | 13.9760 | 15.3420 | 2886.254 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.1560 | 14.7200 | 16.1520 | 2767.422 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.8588 | 13.7284 | 14.0242 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.8686 | 13.6496 | 13.9226 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 14.0587 | 13.6640 | 14.4307 | 6.992 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7153 | 13.3960 | 14.3200 | 7.167 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.9073 | 13.5087 | 14.6460 | 46.847 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.2120 | 13.5473 | 14.4813 | 45.842 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.6540 | 13.2420 | 14.7840 | 307.185 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.5380 | 13.1020 | 15.2060 | 309.817 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.5900 | 13.8800 | 16.7880 | 2263.889 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.4340 | 13.6240 | 15.9880 | 2288.357 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.8600 | 13.8520 | 15.9240 | 2822.546 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.8600 | 13.7380 | 15.7020 | 2822.546 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

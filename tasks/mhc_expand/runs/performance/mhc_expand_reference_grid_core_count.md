# MhcExpand local A3 performance — reference_grid_core_count

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2570 | 13.1814 | 13.8696 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1864 | 12.9880 | 13.5252 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.3180 | 13.0060 | 14.2533 | 7.381 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5247 | 12.9413 | 13.9687 | 7.268 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8493 | 13.1307 | 14.1047 | 47.043 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6987 | 13.1127 | 14.0880 | 47.560 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4760 | 14.2680 | 15.7660 | 2281.718 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.3640 | 14.1220 | 14.8040 | 2299.509 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.0860 | 13.5760 | 16.7520 | 2977.640 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.7720 | 14.3660 | 16.5240 | 2839.361 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.5224 | 13.2670 | 13.7432 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2574 | 13.0154 | 13.5874 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.1360 | 12.9153 | 13.7640 | 7.484 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4133 | 12.8893 | 13.6173 | 7.329 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.7067 | 13.1747 | 14.2893 | 47.532 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7607 | 13.2253 | 14.3800 | 47.346 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.2160 | 13.3020 | 17.1660 | 295.041 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.4480 | 13.8480 | 16.2620 | 290.303 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.8220 | 14.3120 | 16.4180 | 2228.454 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.6500 | 14.3800 | 16.1880 | 2254.617 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.0820 | 14.0080 | 17.7180 | 2781.000 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.5780 | 14.4220 | 17.0720 | 2877.146 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

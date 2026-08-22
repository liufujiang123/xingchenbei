# MhcExpand local A3 performance — compile_m_baseline_ab

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.4208 | 13.3230 | 15.1354 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2936 | 12.9424 | 13.4132 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4453 | 13.1627 | 14.2953 | 7.311 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.7173 | 13.2813 | 14.3700 | 7.166 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.7860 | 13.1527 | 14.3033 | 47.259 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6547 | 13.2527 | 14.1973 | 47.713 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5900 | 14.2220 | 16.2280 | 2263.889 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.9560 | 14.8020 | 16.4720 | 2208.488 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.5340 | 13.5200 | 15.6620 | 2700.080 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.6340 | 15.0320 | 17.0020 | 2682.809 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 14.3372 | 13.8646 | 14.9084 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.9134 | 13.7730 | 14.2332 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 14.3693 | 13.6920 | 14.8913 | 6.841 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 14.0507 | 13.3847 | 14.5207 | 6.996 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 15.1807 | 14.3360 | 15.4507 | 42.917 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 16.2013 | 14.6313 | 17.7173 | 40.213 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1140 | 13.9120 | 15.4420 | 297.173 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.8540 | 12.9880 | 15.2940 | 302.750 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.2420 | 13.6820 | 15.4840 | 2319.207 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.7080 | 14.2740 | 15.7740 | 2245.726 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.9240 | 14.3680 | 15.8820 | 2810.442 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.7740 | 14.4180 | 16.0260 | 2838.976 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

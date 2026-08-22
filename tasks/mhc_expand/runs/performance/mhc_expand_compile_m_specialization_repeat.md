# MhcExpand local A3 performance — compile_m_specialization_repeat

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1136 | 12.8410 | 13.6406 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9820 | 12.7952 | 13.1966 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0167 | 12.7940 | 13.6813 | 7.552 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1500 | 12.8700 | 13.7640 | 7.476 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.6213 | 13.0540 | 13.8307 | 47.830 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6127 | 13.1493 | 13.7747 | 47.861 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5160 | 14.3300 | 16.0780 | 2275.430 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1600 | 13.7280 | 14.8780 | 2332.637 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.9440 | 13.1800 | 16.4580 | 2806.681 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.6000 | 14.3840 | 16.6380 | 2872.811 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1152 | 12.9364 | 13.5032 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9734 | 12.8814 | 13.1992 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6547 | 13.0867 | 13.9833 | 7.199 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4340 | 13.0507 | 13.6313 | 7.318 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.5400 | 13.0860 | 13.7553 | 48.117 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8253 | 12.9787 | 14.4087 | 47.124 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.2020 | 14.1260 | 15.6360 | 295.332 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.2020 | 14.0920 | 15.7500 | 295.332 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 16.3840 | 14.9560 | 20.4100 | 2016.000 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.7500 | 14.5940 | 15.7400 | 2239.332 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.6480 | 14.4620 | 16.0380 | 2863.397 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.7700 | 14.5980 | 15.8520 | 2839.745 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

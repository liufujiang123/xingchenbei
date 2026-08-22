# MhcExpand local A3 performance — mode_compile_specialization

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2148 | 12.8226 | 13.6248 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9394 | 12.7768 | 13.1408 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0207 | 12.5553 | 13.7213 | 7.550 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.2800 | 12.6200 | 13.5693 | 7.402 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4127 | 13.0873 | 13.9753 | 48.574 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.4940 | 12.8500 | 13.9707 | 48.281 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.6120 | 14.1480 | 15.8440 | 2260.481 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.4660 | 12.8700 | 14.6020 | 2452.855 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.5900 | 14.4740 | 16.0480 | 2874.780 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5600 | 14.3920 | 15.7240 | 2880.703 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1190 | 12.8366 | 13.3066 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.1630 | 13.0270 | 13.8302 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.4407 | 12.7860 | 13.9773 | 7.314 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.3160 | 12.9200 | 13.6473 | 7.382 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3293 | 12.7940 | 13.9287 | 48.878 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.5960 | 12.7813 | 13.8147 | 47.919 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.0020 | 13.9100 | 15.7340 | 299.550 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.9020 | 13.6440 | 16.1640 | 301.705 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.9620 | 13.3040 | 15.4580 | 2365.717 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.9040 | 14.3660 | 15.7760 | 2216.193 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.8520 | 14.5400 | 16.0740 | 2824.067 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 13.7160 | 12.8720 | 16.1680 | 3057.964 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

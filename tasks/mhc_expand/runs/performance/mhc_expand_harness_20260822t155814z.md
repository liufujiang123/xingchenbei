# MhcExpand local A3 performance — harness_20260822t155814z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1412 | 12.8864 | 13.4652 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.6424 | 12.5278 | 12.9616 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.7867 | 12.7213 | 12.9860 | 7.688 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1380 | 12.8880 | 13.5740 | 7.482 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.2833 | 13.0420 | 13.8073 | 49.047 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.0593 | 12.7247 | 13.3373 | 49.888 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.5940 | 11.9040 | 15.0460 | 2429.759 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.9120 | 13.4220 | 14.5920 | 2374.220 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.4560 | 14.3280 | 14.7000 | 2901.428 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5560 | 14.3060 | 14.7280 | 2881.495 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.7140 | 12.5480 | 12.9112 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5342 | 12.4516 | 12.8644 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.8693 | 12.6240 | 13.0393 | 7.639 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.8787 | 12.7073 | 12.9913 | 7.633 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.0100 | 12.8100 | 13.0453 | 50.078 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.0547 | 12.9253 | 13.6467 | 49.906 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.8160 | 13.5380 | 14.5520 | 303.583 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.7700 | 13.6420 | 13.9740 | 304.597 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.3700 | 13.6920 | 14.9740 | 2298.549 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.9800 | 13.6020 | 14.7640 | 2362.671 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.0640 | 13.0380 | 22.3880 | 2982.298 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 19.8140 | 12.8400 | 34.4860 | 2116.839 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

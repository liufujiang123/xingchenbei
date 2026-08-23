# MhcExpand local A3 performance — harness_20260822t172525z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3806 | 13.0804 | 14.0964 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2376 | 12.9286 | 13.4282 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5060 | 12.9587 | 14.3160 | 7.279 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4413 | 12.7067 | 14.5840 | 7.314 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4973 | 12.7180 | 14.0800 | 48.270 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.5833 | 12.8093 | 14.0233 | 47.964 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4580 | 11.9220 | 15.7240 | 2284.558 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6320 | 13.9960 | 16.4060 | 2257.391 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.5120 | 13.8380 | 14.7200 | 2890.232 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.9040 | 14.0480 | 17.3120 | 2814.214 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3284 | 13.0232 | 13.6994 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.3136 | 13.1274 | 13.4138 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.4967 | 13.1633 | 14.1687 | 7.284 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.3860 | 12.9987 | 14.0127 | 7.344 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4440 | 12.2047 | 13.8980 | 48.461 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9300 | 13.3793 | 14.1593 | 46.770 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.5500 | 14.2620 | 16.4020 | 288.268 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3380 | 14.1540 | 15.5140 | 292.531 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.7720 | 14.6880 | 17.8420 | 2235.997 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.8220 | 14.3760 | 15.5800 | 2228.454 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.5040 | 14.1580 | 16.4120 | 2891.826 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 15.4060 | 14.8940 | 16.3280 | 2722.513 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — harness_20260822t172248z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 14.3006 | 13.6610 | 14.7892 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.5680 | 13.4518 | 13.8666 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.3500 | 13.0480 | 14.4367 | 7.364 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.6547 | 12.9080 | 14.0400 | 7.199 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4487 | 12.8840 | 14.2233 | 48.444 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3180 | 13.0887 | 14.2160 | 48.920 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.1400 | 13.8420 | 15.9580 | 2335.937 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.3200 | 13.3080 | 15.1520 | 2306.574 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.6560 | 14.4900 | 15.9300 | 2861.834 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.3300 | 13.7580 | 15.6120 | 2926.939 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.4012 | 13.1712 | 13.7608 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.3192 | 13.1470 | 13.4138 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.5167 | 12.6827 | 14.0733 | 7.273 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2107 | 12.6980 | 13.8900 | 7.441 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3673 | 13.2513 | 13.7787 | 48.739 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.2087 | 13.2747 | 14.4100 | 45.853 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1140 | 13.9780 | 15.2220 | 297.173 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.0640 | 13.6880 | 15.9540 | 298.230 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.3940 | 14.1240 | 15.7340 | 2294.716 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.9460 | 13.5140 | 15.0060 | 2368.431 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.4860 | 14.1460 | 16.3040 | 2895.419 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.4860 | 14.3040 | 16.2580 | 2895.419 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

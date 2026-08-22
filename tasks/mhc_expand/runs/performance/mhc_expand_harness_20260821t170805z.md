# MhcExpand local A3 performance — harness_20260821t170805z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.0542 | 12.8042 | 13.5646 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0770 | 12.6376 | 13.1644 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9027 | 12.7640 | 13.3633 | 7.619 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.9920 | 12.7533 | 13.9513 | 7.567 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.6007 | 12.6127 | 14.0780 | 47.903 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.5360 | 13.0227 | 14.1840 | 48.132 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 15.6520 | 14.0540 | 16.3340 | 2110.283 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6000 | 14.1800 | 15.7940 | 2262.339 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.6300 | 14.3460 | 15.3480 | 2866.920 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.8040 | 15.6040 | 16.1580 | 2653.951 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9940 | 12.9182 | 13.3052 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9282 | 12.8012 | 13.5568 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.3033 | 12.7433 | 13.7627 | 7.389 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.0893 | 13.0300 | 13.5173 | 7.510 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3407 | 13.0447 | 14.0260 | 48.836 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8367 | 12.9353 | 14.3540 | 47.086 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.2020 | 12.7980 | 14.6780 | 317.702 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 12.9100 | 12.2380 | 14.6200 | 324.888 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 19.9560 | 19.8340 | 21.0020 | 1655.149 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 21.3220 | 21.2460 | 21.3780 | 1549.111 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 20.4240 | 20.3880 | 20.5240 | 2053.615 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 20.4300 | 20.3840 | 21.3700 | 2053.012 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

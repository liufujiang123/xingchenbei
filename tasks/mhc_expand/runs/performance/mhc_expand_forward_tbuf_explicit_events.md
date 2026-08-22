# MhcExpand local A3 performance — forward_tbuf_explicit_events

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.4696 | 13.1646 | 14.0598 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3190 | 13.1660 | 13.5494 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.8387 | 13.3920 | 14.3840 | 7.104 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 14.1493 | 13.4520 | 14.2733 | 6.948 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 14.1813 | 13.3080 | 14.5647 | 45.941 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.9007 | 13.5907 | 14.3360 | 46.869 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.6120 | 14.2140 | 16.4760 | 2260.481 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6900 | 14.1200 | 14.9040 | 2248.478 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.8120 | 14.1500 | 15.0300 | 2831.693 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4920 | 14.2400 | 15.7000 | 2894.220 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3738 | 12.9590 | 13.5074 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2060 | 13.0614 | 13.8792 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.9433 | 13.5040 | 14.1187 | 7.050 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7933 | 13.7100 | 14.1167 | 7.127 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 14.1680 | 13.8287 | 14.9027 | 45.985 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.2393 | 13.6893 | 14.8707 | 45.754 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.6500 | 14.4820 | 15.7220 | 286.301 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.6320 | 13.7100 | 15.8340 | 286.653 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.7520 | 14.4020 | 15.9080 | 2239.028 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.1020 | 14.9400 | 15.9840 | 2187.137 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.2300 | 14.5880 | 16.3300 | 2753.975 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 15.1860 | 14.6960 | 18.8940 | 2761.954 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

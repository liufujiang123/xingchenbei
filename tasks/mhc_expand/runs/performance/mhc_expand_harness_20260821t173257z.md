# MhcExpand local A3 performance — harness_20260821t173257z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3172 | 13.1436 | 14.1342 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3260 | 13.0396 | 13.6362 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.6053 | 13.0640 | 14.0727 | 7.225 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5667 | 13.2020 | 14.0300 | 7.246 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8673 | 13.1453 | 14.7140 | 46.982 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.7320 | 13.3393 | 14.0620 | 47.445 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.2220 | 14.2060 | 17.0200 | 2322.468 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.9280 | 13.7720 | 16.6060 | 2371.492 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.9940 | 13.6160 | 15.0360 | 2997.216 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.6960 | 14.3880 | 15.8620 | 2854.045 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3486 | 13.0756 | 13.7126 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2200 | 13.1826 | 13.4998 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.8467 | 13.2647 | 13.9007 | 7.099 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7567 | 13.4427 | 14.4693 | 7.146 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.9573 | 13.6113 | 14.1027 | 46.679 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.2613 | 13.6453 | 14.3680 | 45.684 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.5540 | 14.0520 | 15.6420 | 288.189 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 15.1080 | 14.7520 | 16.5440 | 277.621 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 15.9360 | 14.5380 | 16.4680 | 2072.675 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.2300 | 15.0340 | 18.6480 | 2168.755 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.6220 | 14.4200 | 17.3980 | 2684.870 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 15.2880 | 14.8980 | 19.9760 | 2743.527 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

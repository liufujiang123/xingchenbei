# MhcExpand local A3 performance — candidate1_tpipe

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1078 | 12.9718 | 13.6712 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3348 | 12.9944 | 13.5072 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2973 | 12.7573 | 13.8513 | 7.393 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4433 | 12.9273 | 13.9480 | 7.312 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.4900 | 12.8733 | 14.4367 | 48.296 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3367 | 12.9220 | 13.8767 | 48.851 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.6420 | 13.2720 | 14.6140 | 2421.210 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.6860 | 13.5100 | 14.5880 | 2413.426 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.6260 | 14.4400 | 15.1260 | 2867.704 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.6300 | 14.3960 | 16.1120 | 2683.496 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3068 | 12.9204 | 13.5248 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0504 | 12.9438 | 13.7142 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.5673 | 13.1547 | 13.7407 | 7.246 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2000 | 12.7520 | 15.3700 | 7.447 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.2240 | 12.7900 | 13.4633 | 49.267 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6660 | 12.9660 | 14.2327 | 47.674 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 24.6200 | 24.2640 | 26.3800 | 1341.598 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 26.0420 | 25.8180 | 26.2140 | 1268.341 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 30.0820 | 29.9040 | 31.1160 | 1394.290 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 30.1560 | 29.8820 | 30.3660 | 1390.869 |

## Summary

- Cases: 20
- Forward cases: 10
- Backward cases: 10
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

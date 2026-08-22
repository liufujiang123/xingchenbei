# MhcExpand local A3 performance — baseline2

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3926 | 13.2034 | 14.2988 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2682 | 12.6612 | 14.6068 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0020 | 12.8087 | 14.3580 | 7.561 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4820 | 12.8820 | 14.1387 | 7.291 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.5513 | 12.8700 | 14.3147 | 48.077 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.2520 | 12.9787 | 14.5513 | 49.163 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.2340 | 13.9640 | 16.6300 | 2320.510 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.2080 | 13.8480 | 14.7180 | 2324.757 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.5560 | 15.4880 | 15.6340 | 2696.261 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.4780 | 15.4100 | 16.2160 | 2709.849 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3568 | 12.8436 | 13.6284 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2298 | 12.9994 | 13.4118 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6320 | 12.7547 | 14.2987 | 7.211 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5033 | 12.9947 | 13.8560 | 7.280 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3967 | 13.1220 | 13.6727 | 48.632 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.0213 | 13.1147 | 14.4860 | 46.466 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 25.7720 | 24.7800 | 26.8520 | 1281.629 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 24.6500 | 24.5080 | 24.7420 | 1339.965 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.7320 | 29.6420 | 30.5540 | 1410.704 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.6900 | 29.5860 | 30.6720 | 1412.699 |

## Summary

- Cases: 20
- Forward cases: 10
- Backward cases: 10
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

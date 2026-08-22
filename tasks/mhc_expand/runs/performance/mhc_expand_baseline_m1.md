# MhcExpand local A3 performance — baseline_m1

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.5384 | 13.2896 | 13.8956 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.4502 | 13.2078 | 13.5790 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5147 | 13.3120 | 13.7773 | 7.274 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.8300 | 13.4707 | 14.0127 | 7.108 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 14.0533 | 13.4080 | 14.2447 | 46.360 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6647 | 13.5627 | 14.4620 | 47.678 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.7280 | 14.4180 | 17.2300 | 2242.677 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.5580 | 14.1380 | 15.2600 | 2268.866 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.2700 | 15.1860 | 15.5200 | 2746.761 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.3320 | 15.2220 | 16.5380 | 2735.654 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3990 | 13.1510 | 13.6048 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.3142 | 13.1798 | 13.5620 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6993 | 13.2013 | 14.1280 | 7.176 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.6000 | 13.2473 | 13.9180 | 7.228 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.5580 | 13.4327 | 13.9100 | 48.054 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9820 | 13.4133 | 14.1407 | 46.596 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.9200 | 13.6240 | 15.3860 | 301.315 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3840 | 13.4660 | 15.2040 | 291.595 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 25.6580 | 25.5520 | 26.5380 | 1287.323 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 24.2480 | 24.1720 | 24.3600 | 1362.180 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 29.9260 | 29.7100 | 30.1660 | 1401.559 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 31.3720 | 31.0880 | 31.4720 | 1336.958 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

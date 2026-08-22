# MhcExpand local A3 performance — candidate2_first_acc

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 14.1584 | 13.8894 | 14.5614 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.4302 | 13.1162 | 14.2298 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5760 | 12.9400 | 14.2287 | 7.241 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.3987 | 13.2020 | 14.6187 | 7.337 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8200 | 13.1973 | 14.5113 | 47.143 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.9747 | 13.3527 | 14.9000 | 46.621 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3480 | 14.1360 | 17.3580 | 2302.073 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.7080 | 14.1260 | 15.4420 | 2245.726 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.9660 | 13.7840 | 14.8160 | 3003.225 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.0740 | 14.9760 | 16.1920 | 2782.476 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.5534 | 13.1530 | 13.6840 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2852 | 13.1344 | 13.5672 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.7753 | 13.0707 | 14.7527 | 7.136 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7140 | 13.3353 | 14.6913 | 7.168 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6240 | 13.1340 | 14.0453 | 47.821 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 14.0067 | 13.0933 | 14.3147 | 46.514 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 24.3420 | 24.1380 | 24.6900 | 1356.920 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 22.9900 | 22.9200 | 24.1560 | 1436.718 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 27.7480 | 27.6360 | 29.1180 | 1511.570 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.2000 | 29.0220 | 29.3180 | 1436.406 |

## Summary

- Cases: 20
- Forward cases: 10
- Backward cases: 10
- Timing excludes tensor allocation and D2H copy; ACLNN executor creation occurs on the host and is not represented as device work.

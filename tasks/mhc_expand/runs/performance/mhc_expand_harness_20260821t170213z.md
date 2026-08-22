# MhcExpand local A3 performance — harness_20260821t170213z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1808 | 13.0804 | 14.3158 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.2266 | 12.8730 | 13.2994 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5473 | 13.0933 | 13.9460 | 7.256 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.6453 | 13.0900 | 13.7687 | 7.204 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8280 | 13.0747 | 14.1420 | 47.115 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6673 | 13.1300 | 13.8587 | 47.669 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4720 | 14.1320 | 16.2140 | 2282.348 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.2360 | 13.8820 | 15.6700 | 2320.184 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2940 | 14.1860 | 15.9800 | 2934.311 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5400 | 14.3240 | 15.2760 | 2884.666 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2252 | 13.0442 | 14.0312 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0482 | 13.0132 | 13.2726 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6013 | 13.1800 | 13.7993 | 7.228 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5060 | 13.1693 | 13.8193 | 7.279 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.5627 | 13.2160 | 14.1280 | 48.037 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9080 | 13.1907 | 14.1173 | 46.844 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.4520 | 13.5120 | 15.8520 | 290.223 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.5140 | 13.9180 | 26.2360 | 288.983 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 24.8320 | 24.3860 | 25.5980 | 1330.144 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 24.9120 | 24.7520 | 25.0060 | 1325.873 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 27.9780 | 27.6620 | 28.9120 | 1499.144 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 29.2220 | 27.9340 | 29.3360 | 1435.324 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

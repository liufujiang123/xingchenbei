# MhcExpand local A3 performance — harness_20260822t171149z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 12.8156 | 12.6298 | 13.5720 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.6032 | 12.5650 | 12.8844 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9513 | 12.7573 | 13.3940 | 7.590 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.9780 | 12.8980 | 13.1893 | 7.575 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 12.8120 | 12.5273 | 13.4640 | 50.852 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.7873 | 12.4927 | 13.1753 | 50.950 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.2640 | 14.0780 | 14.4720 | 2315.630 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.0580 | 12.8160 | 13.4560 | 2529.495 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.3900 | 13.3000 | 13.6760 | 3132.415 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 13.6260 | 13.2720 | 13.9220 | 3078.162 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.6370 | 12.4354 | 12.8216 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5036 | 12.3356 | 12.7020 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.8500 | 12.7460 | 12.9893 | 7.650 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.8980 | 12.6133 | 13.2293 | 7.622 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.6233 | 12.4033 | 12.6880 | 51.612 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 12.6147 | 12.2953 | 13.0267 | 51.647 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 12.9000 | 12.6620 | 14.2720 | 325.140 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 12.9840 | 12.6040 | 13.3540 | 323.036 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 13.9760 | 13.8740 | 14.2880 | 2363.347 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.9460 | 13.7380 | 15.8980 | 2368.431 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.2800 | 13.6300 | 15.0520 | 2937.188 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.2920 | 13.7120 | 14.5820 | 2934.721 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

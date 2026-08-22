# MhcExpand local A3 performance — backward_row_batch_repeat1

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.5904 | 13.1484 | 16.3776 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0214 | 12.9282 | 14.1956 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.1420 | 9.5087 | 13.4233 | 7.480 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.6753 | 12.8413 | 14.4127 | 7.188 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.7393 | 12.9767 | 14.8040 | 47.419 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6200 | 13.0160 | 13.9327 | 47.835 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3580 | 14.1600 | 16.3360 | 2300.470 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.3220 | 12.9640 | 13.7420 | 2479.368 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 15.3500 | 13.8740 | 15.4140 | 2732.446 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.4280 | 15.1840 | 16.5740 | 2718.631 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.0624 | 12.8114 | 13.5048 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9666 | 12.8156 | 13.4840 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.2153 | 12.7000 | 13.8240 | 7.439 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.0153 | 12.6593 | 13.4733 | 7.553 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.2040 | 12.7240 | 13.9587 | 49.342 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9107 | 12.9433 | 18.2347 | 46.835 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1200 | 13.2800 | 16.2300 | 297.047 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.6980 | 5.6540 | 15.2060 | 306.198 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 19.6060 | 19.4000 | 20.8180 | 1684.696 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 21.0080 | 20.9080 | 21.8660 | 1572.265 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 20.0140 | 19.9200 | 20.1600 | 2095.685 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 20.1220 | 20.0740 | 20.2300 | 2084.437 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

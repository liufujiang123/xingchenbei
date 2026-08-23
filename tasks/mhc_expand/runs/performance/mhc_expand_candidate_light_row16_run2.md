# MhcExpand local A3 performance — candidate_light_row16_run2

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2582 | 13.0382 | 13.8928 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0702 | 12.8314 | 13.3324 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 12.9980 | 12.4447 | 13.6020 | 7.563 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1393 | 12.7220 | 13.8007 | 7.482 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.2787 | 12.7647 | 13.5067 | 49.064 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.9953 | 12.9013 | 14.1907 | 50.134 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.8040 | 13.0320 | 14.8040 | 2392.795 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.0740 | 11.9300 | 17.4520 | 2346.891 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1120 | 13.9380 | 14.6900 | 2972.154 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4020 | 14.1220 | 15.8500 | 2912.307 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1932 | 12.8498 | 13.2652 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9742 | 12.8776 | 13.5140 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.5487 | 12.9233 | 14.3500 | 7.256 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5133 | 13.2473 | 14.0460 | 7.275 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3420 | 13.0633 | 14.1620 | 48.832 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6807 | 13.2373 | 13.8740 | 47.623 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.5020 | 14.2160 | 15.8700 | 289.222 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3740 | 13.9140 | 15.4180 | 291.798 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.3280 | 13.8920 | 15.1500 | 2305.286 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.4960 | 14.1340 | 15.4400 | 2278.570 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.8200 | 14.3220 | 16.0520 | 2830.165 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.9100 | 14.6140 | 15.8620 | 2813.081 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — next_round_baseline_after

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.5698 | 13.2156 | 14.1620 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3416 | 13.1480 | 13.5584 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.8847 | 13.1633 | 14.5347 | 7.080 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5347 | 13.2640 | 14.1753 | 7.263 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 14.1113 | 13.2140 | 14.5433 | 46.169 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.6700 | 13.3633 | 14.2887 | 47.660 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.7120 | 13.8200 | 17.8660 | 2245.116 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1380 | 13.6900 | 14.6120 | 2336.267 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.8360 | 14.0640 | 16.0500 | 2827.112 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 15.1340 | 14.6020 | 16.6440 | 2771.445 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.5332 | 13.2838 | 13.7104 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.5626 | 13.4032 | 14.0294 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.6687 | 13.2367 | 14.2387 | 7.192 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.7960 | 13.3880 | 14.0840 | 7.126 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.9133 | 13.5427 | 14.0787 | 46.826 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9647 | 13.4647 | 14.5073 | 46.654 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.6200 | 14.0400 | 16.5540 | 286.888 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.4420 | 14.0240 | 16.8140 | 290.424 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 15.9700 | 14.8060 | 16.8280 | 2068.262 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.5200 | 15.1460 | 16.7580 | 2128.231 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.4120 | 15.3620 | 17.6120 | 2721.453 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 15.3760 | 15.0600 | 18.1560 | 2727.825 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

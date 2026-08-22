# MhcExpand local A3 performance — reference_grid_core_count_repeat

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2842 | 13.0230 | 14.1978 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1478 | 13.0368 | 13.5728 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.5567 | 13.1727 | 13.9093 | 7.251 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.7113 | 13.0920 | 13.9467 | 7.170 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8533 | 13.2613 | 14.2593 | 47.029 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.8073 | 13.2780 | 13.9787 | 47.186 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.5820 | 14.5460 | 15.9380 | 2265.131 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.3540 | 14.1340 | 14.8680 | 2301.111 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.4360 | 14.0780 | 14.6900 | 2905.447 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5400 | 14.3360 | 15.3560 | 2884.666 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2526 | 13.1578 | 13.5568 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0560 | 12.4704 | 13.3892 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.3760 | 13.0567 | 13.5500 | 7.349 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.4773 | 12.9693 | 13.7613 | 7.294 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4973 | 13.2447 | 13.8147 | 48.270 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.8653 | 12.6527 | 14.6300 | 46.988 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.3840 | 14.1500 | 15.6700 | 291.595 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.9920 | 14.3120 | 18.6380 | 279.769 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.7760 | 14.6660 | 16.1580 | 2235.391 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.8820 | 14.7620 | 15.8160 | 2219.469 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.9120 | 14.4460 | 16.2780 | 2812.704 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.8400 | 14.4820 | 16.1620 | 2826.351 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

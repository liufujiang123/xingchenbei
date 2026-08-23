# MhcExpand local A3 performance — harness_20260822t154219z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2770 | 12.9036 | 14.0100 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9488 | 12.7274 | 13.2126 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0940 | 12.6387 | 14.2747 | 7.508 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4000 | 12.5953 | 14.2340 | 7.336 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.1707 | 12.6533 | 13.9227 | 49.467 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.9353 | 12.4513 | 14.2427 | 50.367 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.1120 | 13.9200 | 16.6360 | 2340.571 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.5660 | 13.3180 | 14.5820 | 2434.774 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.1600 | 13.3500 | 14.3200 | 2962.079 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4500 | 13.5120 | 15.7640 | 2902.632 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3952 | 13.0974 | 13.6532 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0898 | 12.8796 | 13.4402 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.3507 | 12.6060 | 14.4060 | 7.363 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.9500 | 12.7327 | 14.0920 | 7.591 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.0953 | 12.8847 | 13.5560 | 49.751 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7220 | 12.7873 | 14.0613 | 47.479 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.2980 | 13.0200 | 15.6020 | 315.409 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.0560 | 13.2320 | 15.7360 | 298.400 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.8040 | 14.4260 | 15.9340 | 2231.164 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.3080 | 13.4300 | 14.7000 | 2308.509 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.5280 | 14.0680 | 16.4060 | 2887.048 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.1080 | 13.6000 | 15.4620 | 2972.997 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

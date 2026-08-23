# MhcExpand local A3 performance — harness_20260822t173200z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.0642 | 12.7296 | 13.8898 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.7134 | 12.6362 | 13.0456 | 0.008 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.0827 | 12.6320 | 13.3853 | 7.514 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 12.7553 | 12.3987 | 12.9440 | 7.707 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 12.8560 | 12.4013 | 13.3873 | 50.678 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.7373 | 12.2240 | 12.9360 | 51.150 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 13.8600 | 12.8420 | 14.3640 | 2383.127 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.4660 | 12.8060 | 14.3980 | 2452.855 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2400 | 13.4260 | 15.4780 | 2945.438 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5900 | 14.3200 | 15.4940 | 2874.780 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.6646 | 12.4160 | 12.9532 | 0.008 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.5492 | 12.4404 | 13.0824 | 0.008 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 12.6467 | 12.5133 | 12.7273 | 7.773 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 12.6447 | 12.6307 | 12.8173 | 7.774 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 12.7000 | 12.5160 | 13.6767 | 51.300 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 12.5273 | 12.4720 | 12.9293 | 52.007 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 12.9940 | 12.2500 | 13.8520 | 322.788 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 12.5080 | 12.2040 | 13.4680 | 335.330 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 15.9900 | 14.8280 | 16.3860 | 2065.675 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 13.5860 | 13.4000 | 14.1360 | 2431.190 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.4980 | 14.3420 | 14.7880 | 2893.023 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 13.6080 | 12.9780 | 14.3020 | 3082.234 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — serial_wide_screen

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1844 | 12.8282 | 13.5102 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0250 | 12.8982 | 13.4194 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2420 | 12.7160 | 14.3240 | 7.424 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.1760 | 12.6607 | 14.3480 | 7.461 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.5413 | 12.7460 | 14.5447 | 48.113 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.3447 | 12.8080 | 14.5693 | 48.822 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.0680 | 13.1800 | 15.5380 | 2347.892 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1780 | 12.1620 | 14.6160 | 2329.676 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 13.5480 | 13.1560 | 24.6540 | 3095.884 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.4940 | 13.9300 | 18.5440 | 2893.821 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.2062 | 12.8844 | 13.7766 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0336 | 12.9094 | 13.4242 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.3927 | 12.8087 | 15.6707 | 7.340 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2600 | 12.9073 | 14.4060 | 7.414 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4193 | 13.0867 | 13.8120 | 48.550 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.5187 | 13.1273 | 15.2107 | 48.193 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.6780 | 13.4740 | 15.7300 | 306.646 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.4520 | 13.3640 | 14.5140 | 311.798 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 20.9800 | 20.9340 | 21.6380 | 1574.363 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 21.0460 | 20.9640 | 21.5700 | 1569.426 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.7200 | 14.2580 | 15.8820 | 2849.391 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.5660 | 14.3080 | 15.2420 | 2879.517 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

# MhcExpand local A3 performance — aligned_datacopy_full_repeat

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3932 | 13.1174 | 13.6694 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9056 | 12.7394 | 13.2630 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2307 | 12.8127 | 13.5893 | 7.430 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.5447 | 12.9480 | 13.7133 | 7.258 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.5413 | 12.9880 | 14.2480 | 48.113 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.5333 | 13.0713 | 13.7880 | 48.141 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.0780 | 13.7160 | 15.4300 | 2346.224 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.1700 | 13.8960 | 15.0680 | 2330.991 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.2860 | 14.1260 | 14.9360 | 2935.954 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.2040 | 14.1540 | 15.1640 | 2952.903 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.0572 | 12.9198 | 13.3324 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.0602 | 12.9426 | 13.2584 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.4113 | 13.0347 | 13.8447 | 7.330 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.3087 | 13.0973 | 13.3413 | 7.386 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.4133 | 13.0853 | 14.0873 | 48.572 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.6893 | 13.1780 | 14.3233 | 47.593 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1920 | 13.8280 | 15.0680 | 295.540 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.3280 | 13.8320 | 15.0880 | 292.735 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.5180 | 14.3080 | 15.6620 | 2275.117 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.3160 | 14.1720 | 16.4400 | 2307.219 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.6260 | 14.1900 | 15.6920 | 2867.704 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.4840 | 13.6820 | 17.8660 | 2895.819 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

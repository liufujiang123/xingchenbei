# MhcExpand local A3 performance — harness_20260821t171137z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.2170 | 12.9578 | 13.9030 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.0358 | 12.9170 | 13.3360 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.4667 | 13.2513 | 14.0313 | 7.300 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4860 | 13.1707 | 14.0467 | 7.289 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3653 | 13.1020 | 14.1733 | 48.746 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 12.9180 | 12.7087 | 13.3040 | 50.434 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3460 | 14.0080 | 15.5260 | 2302.394 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 13.5500 | 13.2300 | 15.0400 | 2437.649 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.8420 | 14.3980 | 15.2800 | 2825.969 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.0100 | 13.7500 | 14.7040 | 2993.793 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9300 | 12.6542 | 13.0464 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.8640 | 12.6690 | 13.1642 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.1933 | 13.1440 | 13.8507 | 7.451 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.3020 | 13.1573 | 13.8707 | 7.390 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.1907 | 13.0700 | 13.9207 | 49.392 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.3240 | 13.1333 | 13.7100 | 48.897 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.8880 | 13.3500 | 14.8160 | 302.009 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.8460 | 13.6180 | 15.9380 | 302.925 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 20.0940 | 19.9980 | 20.1640 | 1643.781 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 20.1400 | 20.0700 | 20.2820 | 1640.027 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 20.7380 | 20.5780 | 21.8440 | 2022.521 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 20.6380 | 20.5900 | 21.0020 | 2032.321 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

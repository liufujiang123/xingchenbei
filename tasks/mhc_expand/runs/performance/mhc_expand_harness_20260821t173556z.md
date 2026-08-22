# MhcExpand local A3 performance — harness_20260821t173556z

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1560 | 13.1174 | 13.7154 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.1804 | 12.8360 | 13.3888 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.1007 | 12.5580 | 14.0347 | 7.504 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.4173 | 12.7913 | 13.7487 | 7.327 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.3520 | 12.6860 | 13.6020 | 48.795 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.2600 | 12.8080 | 13.7527 | 49.133 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.3640 | 14.1500 | 16.3620 | 2299.509 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.4680 | 13.9580 | 16.2920 | 2282.979 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.3160 | 13.9520 | 15.8200 | 2929.802 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.5800 | 14.0620 | 16.1920 | 2876.752 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.1646 | 12.9156 | 13.4470 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9734 | 12.9044 | 13.1934 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.2220 | 12.5893 | 14.2327 | 7.435 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.6207 | 13.2900 | 14.5147 | 7.217 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.6113 | 13.2527 | 13.9747 | 47.865 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.9933 | 13.2607 | 14.3160 | 46.559 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 14.1860 | 13.9640 | 15.1420 | 295.665 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 14.1060 | 13.6460 | 15.8720 | 297.342 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 15.7060 | 15.6020 | 16.9580 | 2103.027 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 16.2960 | 16.2200 | 16.5120 | 2026.887 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 15.8240 | 15.7160 | 16.3780 | 2650.596 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 16.1980 | 16.1720 | 70.0860 | 2589.396 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

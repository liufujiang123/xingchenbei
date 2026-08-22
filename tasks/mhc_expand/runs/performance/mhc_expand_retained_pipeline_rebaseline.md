# MhcExpand local A3 performance — retained_pipeline_rebaseline

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.1312 | 12.8154 | 13.6910 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 12.9360 | 12.5988 | 13.2430 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.2673 | 12.7633 | 13.4627 | 7.409 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.0687 | 12.7540 | 13.6593 | 7.522 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.6127 | 12.9547 | 13.6773 | 47.861 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 13.5607 | 13.1987 | 13.9413 | 48.044 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.1000 | 12.1320 | 15.2680 | 2342.563 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.2960 | 12.1440 | 14.9220 | 2310.447 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.7200 | 14.6460 | 14.7480 | 2849.391 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.7600 | 14.5920 | 15.6440 | 2841.669 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 12.9680 | 12.6310 | 14.3886 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 12.9844 | 12.8642 | 13.3636 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.5987 | 13.1007 | 13.7213 | 7.229 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.2987 | 13.1353 | 13.9273 | 7.392 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.3320 | 12.7460 | 14.4747 | 48.868 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.4720 | 12.6780 | 15.0027 | 48.360 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.1740 | 12.9960 | 14.7760 | 318.377 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.4460 | 13.0860 | 14.6400 | 311.937 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 14.9200 | 14.8840 | 16.0900 | 2213.817 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 15.0040 | 14.9060 | 15.7440 | 2201.423 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.9040 | 14.3520 | 37.6720 | 2814.214 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.1240 | 13.6920 | 17.9020 | 2969.629 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

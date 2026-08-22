# MhcExpand local A3 performance — row_tasks_per_core8

> Proxy metric only: CANN 8.5 ACL runtime event time on local Ascend910_9382; not a CANNJudge 910B score.
> Warmup=5; active samples=7; each sample uses case-dependent repeated launches.

| Case | Mode | Shape | DType | m | Median us | Min us | P90 us | Effective GB/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fwd_fp16_smoke | forward | [1, 16] | fp16 | 2 | 13.3386 | 13.0510 | 14.1056 | 0.007 |
| fwd_bf16_smoke | forward | [1, 16] | bf16 | 2 | 13.3914 | 12.9798 | 13.4462 | 0.007 |
| fwd_fp16_published_small | forward | [64, 256] | fp16 | 2 | 13.6707 | 12.9287 | 14.0313 | 7.191 |
| fwd_bf16_published_small | forward | [64, 256] | bf16 | 2 | 13.8173 | 13.0287 | 14.3120 | 7.115 |
| fwd_fp16_nonaligned | forward | [127, 513] | fp16 | 4 | 13.8660 | 13.3860 | 14.4220 | 46.986 |
| fwd_bf16_nonaligned | forward | [127, 513] | bf16 | 4 | 14.0227 | 13.2793 | 14.5047 | 46.461 |
| fwd_fp16_m8_wide | forward | [256, 7168] | fp16 | 8 | 14.4080 | 14.3340 | 15.4920 | 2292.486 |
| fwd_bf16_m8_wide | forward | [256, 7168] | bf16 | 8 | 14.6800 | 14.5860 | 16.9180 | 2250.010 |
| fwd_fp16_published_medium | forward | [1024, 4096] | fp16 | 4 | 14.4180 | 13.7180 | 16.4420 | 2909.075 |
| fwd_bf16_published_medium | forward | [1024, 4096] | bf16 | 4 | 14.8400 | 14.5460 | 17.2940 | 2826.351 |
| bwd_fp16_smoke | backward | [1, 2, 16] | fp16 | 2 | 13.3438 | 13.0566 | 13.5946 | 0.007 |
| bwd_bf16_smoke | backward | [1, 2, 16] | bf16 | 2 | 13.2778 | 13.0212 | 13.5862 | 0.007 |
| bwd_fp16_published_small | backward | [64, 2, 256] | fp16 | 2 | 13.7733 | 13.1567 | 14.2680 | 7.137 |
| bwd_bf16_published_small | backward | [64, 2, 256] | bf16 | 2 | 13.5293 | 13.1807 | 13.9560 | 7.266 |
| bwd_fp16_nonaligned | backward | [127, 4, 513] | fp16 | 4 | 13.1207 | 13.0800 | 13.8533 | 49.655 |
| bwd_bf16_nonaligned | backward | [127, 4, 513] | bf16 | 4 | 13.7967 | 12.9440 | 14.6800 | 47.222 |
| bwd_fp16_m1_medium | backward | [256, 1, 4096] | fp16 | 1 | 13.4780 | 13.2600 | 17.4360 | 311.196 |
| bwd_bf16_m1_medium | backward | [256, 1, 4096] | bf16 | 1 | 13.5040 | 13.0960 | 15.4440 | 310.597 |
| bwd_fp16_m8_wide | backward | [256, 8, 7168] | fp16 | 8 | 16.0360 | 15.7920 | 16.7300 | 2059.750 |
| bwd_bf16_m8_wide | backward | [256, 8, 7168] | bf16 | 8 | 14.9740 | 14.8020 | 16.7920 | 2205.833 |
| bwd_fp16_published_medium | backward | [1024, 4, 4096] | fp16 | 4 | 14.7520 | 13.6560 | 16.0520 | 2843.210 |
| bwd_bf16_published_medium | backward | [1024, 4, 4096] | bf16 | 4 | 14.2400 | 12.7620 | 17.5460 | 2945.438 |

## Summary

- Cases: 22
- Forward cases: 10
- Backward cases: 12
- Timing excludes tensor allocation and D2H copy. The event interval can include stream idle gaps caused by host-side ACLNN executor creation/enqueue pacing; use it only for same-path A/B screening.

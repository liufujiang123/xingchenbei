# MhcSinkhorn local A3 msprof — padded_repeat_n4

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 455
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=22.440000, mean=22.505890, min=22.080000, max=34.620000 |
| Task wait time (us) | median=0.020000, mean=2.290857, min=0.000000, max=206.520000 |
| AIV time (us) | median=20.485000, mean=20.562290, min=20.095000, max=29.886000 |
| Vector pipe time (us) | median=15.267000, mean=15.267004, min=15.267000, max=15.269000 |
| Vector pipe ratio | median=0.745000, mean=0.742725, min=0.511000, max=0.760000 (median 74.500%) |
| Scalar pipe time (us) | median=7.548000, mean=7.578831, min=7.422000, max=8.812000 |
| Scalar pipe ratio | median=0.366000, mean=0.368644, min=0.295000, max=0.405000 (median 36.600%) |
| MTE2 time (us) | median=2.342000, mean=2.328824, min=1.392000, max=3.206000 |
| MTE2 ratio | median=0.114000, mean=0.113211, min=0.069000, max=0.124000 (median 11.400%) |
| MTE3 time (us) | median=1.196000, mean=1.198965, min=1.130000, max=1.875000 |
| MTE3 ratio | median=0.058000, mean=0.058319, min=0.055000, max=0.063000 (median 5.800%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000059, min=0.000000, max=0.027000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

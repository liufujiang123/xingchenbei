# MhcSinkhorn local A3 msprof — retained_baseline

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 185
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=270.780000, mean=270.847676, min=270.440000, max=280.680000 |
| Task wait time (us) | median=0.000000, mean=7.112378, min=0.000000, max=214.980000 |
| AIV time (us) | median=268.758000, mean=268.786281, min=268.317000, max=277.066000 |
| Vector pipe time (us) | median=0.273000, mean=0.272984, min=0.270000, max=0.273000 |
| Vector pipe ratio | median=0.001000, mean=0.001000, min=0.001000, max=0.001000 (median 0.100%) |
| Scalar pipe time (us) | median=267.049000, mean=267.087368, min=266.843000, max=268.893000 |
| Scalar pipe ratio | median=0.994000, mean=0.993654, min=0.971000, max=0.997000 (median 99.400%) |
| MTE2 time (us) | median=1.741000, mean=1.727341, min=0.841000, max=2.521000 |
| MTE2 ratio | median=0.006000, mean=0.006427, min=0.003000, max=0.009000 (median 0.600%) |
| MTE3 time (us) | median=0.488000, mean=0.489005, min=0.488000, max=0.663000 |
| MTE3 ratio | median=0.002000, mean=0.002000, min=0.002000, max=0.002000 (median 0.200%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000011, min=0.000000, max=0.002000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

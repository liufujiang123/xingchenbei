# MhcSinkhorn local A3 msprof — batch32_iter100_conflict

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=26.880000, mean=27.101263, min=26.740000, max=35.520000 |
| Task wait time (us) | median=0.070000, mean=10.780316, min=0.000000, max=215.410000 |
| AIV time (us) | median=25.104000, mean=25.242053, min=24.955000, max=32.049000 |
| Vector pipe time (us) | median=19.883000, mean=19.882979, min=19.881000, max=19.883000 |
| Vector pipe ratio | median=0.792000, mean=0.788211, min=0.620000, max=0.797000 (median 79.200%) |
| Scalar pipe time (us) | median=8.883000, mean=8.913768, min=8.795000, max=10.566000 |
| Scalar pipe ratio | median=0.352000, mean=0.353168, min=0.330000, max=0.369000 (median 35.200%) |
| MTE2 time (us) | median=1.152000, mean=1.159042, min=0.698000, max=1.460000 |
| MTE2 ratio | median=0.046000, mean=0.045853, min=0.028000, max=0.057000 (median 4.600%) |
| MTE3 time (us) | median=0.126000, mean=0.127642, min=0.126000, max=0.282000 |
| MTE3 ratio | median=0.005000, mean=0.005042, min=0.005000, max=0.009000 (median 0.500%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000105, min=0.000000, max=0.010000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

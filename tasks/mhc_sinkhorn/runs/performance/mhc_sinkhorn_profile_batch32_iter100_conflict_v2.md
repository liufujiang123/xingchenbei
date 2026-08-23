# MhcSinkhorn local A3 msprof — batch32_iter100_conflict_v2

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=26.920000, mean=27.116211, min=26.780000, max=35.000000 |
| Task wait time (us) | median=0.080000, mean=10.554842, min=0.000000, max=177.570000 |
| AIV time (us) | median=25.086000, mean=25.220284, min=24.910000, max=31.558000 |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

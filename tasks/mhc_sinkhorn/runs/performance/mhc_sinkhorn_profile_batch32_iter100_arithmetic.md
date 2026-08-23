# MhcSinkhorn local A3 msprof — batch32_iter100_arithmetic

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=27.160000, mean=27.172421, min=26.740000, max=36.240000 |
| Task wait time (us) | median=0.070000, mean=10.197474, min=0.000000, max=164.240000 |
| AIV time (us) | median=25.269000, mean=25.297663, min=24.924000, max=32.455000 |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

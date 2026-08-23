# MhcSinkhorn local A3 msprof — current-iter100

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=39.980000, mean=40.033053, min=39.620000, max=48.740000 |
| Task wait time (us) | median=0.020000, mean=11.702526, min=0.000000, max=218.840000 |
| AIV time (us) | median=38.060000, mean=38.095021, min=37.641000, max=44.840000 |
| Vector pipe time (us) | median=32.229000, mean=32.228979, min=32.227000, max=32.229000 |
| Vector pipe ratio | median=0.847000, mean=0.846316, min=0.719000, max=0.856000 (median 84.700%) |
| Scalar pipe time (us) | median=9.273000, mean=9.280221, min=9.109000, max=11.032000 |
| Scalar pipe ratio | median=0.243000, mean=0.243568, min=0.241000, max=0.264000 (median 24.300%) |
| MTE2 time (us) | median=1.657000, mean=1.616842, min=0.588000, max=1.837000 |
| MTE2 ratio | median=0.043000, mean=0.042421, min=0.016000, max=0.048000 (median 4.300%) |
| MTE3 time (us) | median=0.129000, mean=0.131474, min=0.129000, max=0.364000 |
| MTE3 ratio | median=0.003000, mean=0.003053, min=0.003000, max=0.008000 (median 0.300%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000116, min=0.000000, max=0.011000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

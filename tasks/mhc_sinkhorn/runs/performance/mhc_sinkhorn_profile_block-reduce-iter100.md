# MhcSinkhorn local A3 msprof — block-reduce-iter100

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=26.740000, mean=26.767158, min=26.300000, max=36.260000 |
| Task wait time (us) | median=0.060000, mean=9.936737, min=0.000000, max=178.910000 |
| AIV time (us) | median=24.825000, mean=24.862968, min=24.455000, max=32.019000 |
| Vector pipe time (us) | median=19.439000, mean=19.439053, min=19.439000, max=19.444000 |
| Vector pipe ratio | median=0.783000, mean=0.782411, min=0.607000, max=0.795000 (median 78.300%) |
| Scalar pipe time (us) | median=8.791000, mean=8.892116, min=8.752000, max=10.657000 |
| Scalar pipe ratio | median=0.357000, mean=0.357726, min=0.333000, max=0.382000 (median 35.700%) |
| MTE2 time (us) | median=1.223000, mean=1.201463, min=0.470000, max=1.447000 |
| MTE2 ratio | median=0.049000, mean=0.048316, min=0.019000, max=0.058000 (median 4.900%) |
| MTE3 time (us) | median=0.127000, mean=0.128495, min=0.127000, max=0.269000 |
| MTE3 ratio | median=0.005000, mean=0.005032, min=0.005000, max=0.008000 (median 0.500%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000137, min=0.000000, max=0.013000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

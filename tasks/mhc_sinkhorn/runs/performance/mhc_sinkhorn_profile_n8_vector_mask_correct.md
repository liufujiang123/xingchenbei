# MhcSinkhorn local A3 msprof — n8_vector_mask_correct

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 185
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=22.720000, mean=22.856216, min=22.580000, max=29.780000 |
| Task wait time (us) | median=0.070000, mean=8.507297, min=0.000000, max=250.560000 |
| AIV time (us) | median=20.845000, mean=20.953119, min=20.534000, max=26.348000 |
| Vector pipe time (us) | median=15.518000, mean=15.518005, min=15.518000, max=15.519000 |
| Vector pipe ratio | median=0.744000, mean=0.740838, min=0.589000, max=0.756000 (median 74.400%) |
| Scalar pipe time (us) | median=7.980000, mean=7.987654, min=7.839000, max=9.697000 |
| Scalar pipe ratio | median=0.379000, mean=0.381227, min=0.368000, max=0.411000 (median 37.900%) |
| MTE2 time (us) | median=1.862000, mean=1.891476, min=0.989000, max=2.108000 |
| MTE2 ratio | median=0.089000, mean=0.090227, min=0.048000, max=0.099000 (median 8.900%) |
| MTE3 time (us) | median=0.484000, mean=0.484438, min=0.483000, max=0.642000 |
| MTE3 ratio | median=0.023000, mean=0.023011, min=0.023000, max=0.024000 (median 2.300%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000119, min=0.000000, max=0.022000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

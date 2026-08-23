# MhcSinkhorn local A3 msprof — batch32_iter20

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 185
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=8.920000, mean=9.093081, min=8.760000, max=18.960000 |
| Task wait time (us) | median=10.390000, mean=18.141892, min=0.000000, max=188.100000 |
| AIV time (us) | median=7.209000, mean=7.339005, min=6.968000, max=14.503000 |
| Vector pipe time (us) | median=4.061000, mean=4.060989, min=4.059000, max=4.061000 |
| Vector pipe ratio | median=0.563000, mean=0.555146, min=0.280000, max=0.583000 (median 56.300%) |
| Scalar pipe time (us) | median=2.790000, mean=2.865470, min=2.735000, max=4.884000 |
| Scalar pipe ratio | median=0.387000, mean=0.390735, min=0.337000, max=0.487000 (median 38.700%) |
| MTE2 time (us) | median=1.354000, mean=1.375589, min=0.571000, max=1.687000 |
| MTE2 ratio | median=0.187000, mean=0.187524, min=0.082000, max=0.211000 (median 18.700%) |
| MTE3 time (us) | median=0.127000, mean=0.128351, min=0.127000, max=0.377000 |
| MTE3 ratio | median=0.018000, mean=0.017659, min=0.017000, max=0.026000 (median 1.800%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000227, min=0.000000, max=0.042000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

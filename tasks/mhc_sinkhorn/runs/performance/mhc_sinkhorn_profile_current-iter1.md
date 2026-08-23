# MhcSinkhorn local A3 msprof — current-iter1

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 455
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=5.520000, mean=5.610022, min=5.280000, max=13.640000 |
| Task wait time (us) | median=13.110000, mean=18.134286, min=0.000000, max=246.550000 |
| AIV time (us) | median=3.588000, mean=3.709675, min=3.203000, max=9.879000 |
| Vector pipe time (us) | median=0.549000, mean=0.549002, min=0.549000, max=0.550000 |
| Vector pipe ratio | median=0.153000, mean=0.148851, min=0.056000, max=0.172000 (median 15.300%) |
| Scalar pipe time (us) | median=1.635000, mean=1.694580, min=1.506000, max=3.604000 |
| Scalar pipe ratio | median=0.441000, mean=0.457433, min=0.365000, max=0.732000 (median 44.100%) |
| MTE2 time (us) | median=1.588000, mean=1.587721, min=0.504000, max=1.884000 |
| MTE2 ratio | median=0.438000, mean=0.427960, min=0.141000, max=0.473000 (median 43.800%) |
| MTE3 time (us) | median=0.137000, mean=0.143211, min=0.129000, max=0.411000 |
| MTE3 ratio | median=0.037000, mean=0.038585, min=0.032000, max=0.046000 (median 3.700%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000363, min=0.000000, max=0.165000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

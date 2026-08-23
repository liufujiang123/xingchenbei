# MhcSinkhorn local A3 msprof — batch32_iter100

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 95
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=27.180000, mean=27.211368, min=26.740000, max=35.640000 |
| Task wait time (us) | median=0.070000, mean=11.898632, min=0.000000, max=206.140000 |
| AIV time (us) | median=25.406000, mean=25.422242, min=25.006000, max=32.365000 |
| Vector pipe time (us) | median=19.883000, mean=19.882979, min=19.881000, max=19.883000 |
| Vector pipe ratio | median=0.783000, mean=0.782558, min=0.614000, max=0.795000 (median 78.300%) |
| Scalar pipe time (us) | median=8.776000, mean=8.892347, min=8.675000, max=10.817000 |
| Scalar pipe ratio | median=0.346000, mean=0.349874, min=0.334000, max=0.376000 (median 34.600%) |
| MTE2 time (us) | median=1.502000, mean=1.446653, min=0.662000, max=2.078000 |
| MTE2 ratio | median=0.059000, mean=0.056853, min=0.026000, max=0.064000 (median 5.900%) |
| MTE3 time (us) | median=0.127000, mean=0.129189, min=0.127000, max=0.335000 |
| MTE3 ratio | median=0.005000, mean=0.005053, min=0.005000, max=0.010000 (median 0.500%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000105, min=0.000000, max=0.010000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

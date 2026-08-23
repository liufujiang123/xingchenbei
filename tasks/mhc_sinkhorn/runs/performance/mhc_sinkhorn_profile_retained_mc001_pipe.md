# MhcSinkhorn local A3 msprof — retained_mc001_pipe

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 905
- Task type: AI_VECTOR_CORE
- Exported block dimension: 1

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=4.180000, mean=4.293591, min=4.080000, max=7.620000 |
| Task wait time (us) | median=15.400000, mean=16.329094, min=0.000000, max=157.980000 |
| AIV time (us) | median=3.664000, mean=3.772697, min=3.552000, max=7.074000 |
| Vector pipe time (us) | median=2.401000, mean=2.400894, min=2.399000, max=2.401000 |
| Vector pipe ratio | median=0.655000, mean=0.643739, min=0.339000, max=0.676000 (median 65.500%) |
| Scalar pipe time (us) | median=1.706000, mean=1.762845, min=1.619000, max=3.213000 |
| Scalar pipe ratio | median=0.466000, mean=0.467183, min=0.435000, max=0.536000 (median 46.600%) |
| MTE2 time (us) | median=0.204000, mean=0.223269, min=0.156000, max=1.075000 |
| MTE2 ratio | median=0.056000, mean=0.057513, min=0.037000, max=0.156000 (median 5.600%) |
| MTE3 time (us) | median=0.125000, mean=0.125181, min=0.099000, max=0.168000 |
| MTE3 ratio | median=0.033000, mean=0.033465, min=0.017000, max=0.041000 (median 3.300%) |
| AIV I-cache miss rate | median=0.000000, mean=0.003295, min=0.000000, max=0.070000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 1.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

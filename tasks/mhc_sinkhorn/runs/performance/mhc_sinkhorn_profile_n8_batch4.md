# MhcSinkhorn local A3 msprof — n8_batch4

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 185
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=13.260000, mean=13.297081, min=12.920000, max=22.540000 |
| Task wait time (us) | median=6.790000, mean=15.877459, min=0.000000, max=283.540000 |
| AIV time (us) | median=11.283000, mean=11.323589, min=10.760000, max=18.038000 |
| Vector pipe time (us) | median=7.447000, mean=7.447000, min=7.447000, max=7.447000 |
| Vector pipe ratio | median=0.660000, mean=0.658595, min=0.413000, max=0.692000 (median 66.000%) |
| Scalar pipe time (us) | median=3.492000, mean=3.505973, min=3.271000, max=4.334000 |
| Scalar pipe ratio | median=0.306000, mean=0.309870, min=0.240000, max=0.383000 (median 30.600%) |
| MTE2 time (us) | median=1.575000, mean=1.539319, min=0.463000, max=1.867000 |
| MTE2 ratio | median=0.139000, mean=0.135924, min=0.043000, max=0.161000 (median 13.900%) |
| MTE3 time (us) | median=0.128000, mean=0.129281, min=0.128000, max=0.365000 |
| MTE3 ratio | median=0.011000, mean=0.011292, min=0.011000, max=0.020000 (median 1.100%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000232, min=0.000000, max=0.043000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

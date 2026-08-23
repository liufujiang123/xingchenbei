# MhcSinkhorn local A3 msprof — padded_vector_n4

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 455
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=30.780000, mean=30.798593, min=30.420000, max=39.520000 |
| Task wait time (us) | median=0.000000, mean=3.096176, min=0.000000, max=221.510000 |
| AIV time (us) | median=28.679000, mean=28.766407, min=28.238000, max=34.859000 |
| Vector pipe time (us) | median=22.298000, mean=22.298000, min=22.298000, max=22.298000 |
| Vector pipe ratio | median=0.778000, mean=0.775240, min=0.640000, max=0.790000 (median 77.800%) |
| Scalar pipe time (us) | median=8.396000, mean=8.454314, min=8.271000, max=10.399000 |
| Scalar pipe ratio | median=0.292000, mean=0.293908, min=0.289000, max=0.325000 (median 29.200%) |
| MTE2 time (us) | median=2.448000, mean=2.436084, min=1.358000, max=2.830000 |
| MTE2 ratio | median=0.085000, mean=0.084642, min=0.048000, max=0.097000 (median 8.500%) |
| MTE3 time (us) | median=1.085000, mean=1.086310, min=1.049000, max=1.302000 |
| MTE3 ratio | median=0.038000, mean=0.037752, min=0.036000, max=0.040000 (median 3.800%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000040, min=0.000000, max=0.018000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

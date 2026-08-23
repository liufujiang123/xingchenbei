# MhcSinkhorn local A3 msprof — padded_vector_n6

> Evidence class: `profile_observed`.
> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.

- Profiled MhcSinkhorn tasks: 185
- Task type: AI_VECTOR_CORE
- Exported block dimension: 48

| Exported metric | Aggregate over profiled tasks |
|---|---|
| Task duration (us) | median=40.400000, mean=40.613622, min=40.260000, max=50.320000 |
| Task wait time (us) | median=0.040000, mean=5.866649, min=0.000000, max=222.930000 |
| AIV time (us) | median=38.723000, mean=38.830216, min=38.457000, max=46.163000 |
| Vector pipe time (us) | median=31.447000, mean=31.447000, min=31.447000, max=31.447000 |
| Vector pipe ratio | median=0.812000, mean=0.809968, min=0.681000, max=0.818000 (median 81.200%) |
| Scalar pipe time (us) | median=9.045000, mean=9.078395, min=8.917000, max=11.287000 |
| Scalar pipe ratio | median=0.233000, mean=0.233795, min=0.230000, max=0.247000 (median 23.300%) |
| MTE2 time (us) | median=2.368000, mean=2.381092, min=1.591000, max=3.120000 |
| MTE2 ratio | median=0.061000, mean=0.061314, min=0.041000, max=0.069000 (median 6.100%) |
| MTE3 time (us) | median=1.548000, mean=1.551816, min=1.546000, max=1.848000 |
| MTE3 ratio | median=0.040000, mean=0.040000, min=0.040000, max=0.040000 (median 4.000%) |
| AIV I-cache miss rate | median=0.000000, mean=0.000092, min=0.000000, max=0.017000 (median 0.000%) |

## Interpretation boundary

- Observed block dimensions: 48.
- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.
- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.
- No unexported stall, bandwidth, or utilization metric is inferred here.

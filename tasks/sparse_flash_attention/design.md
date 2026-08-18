# SparseFlashAttention design

## Baseline semantics

TODO after importing the authoritative problem statement.

## Dataflow

TODO.

## Host Tiling

TODO.

## Memory plan

TODO.

## Core partition

TODO.

## Precision risks

TODO.

## Optimization candidates

- sparse gather aggregation / locality;
- Q/head/sparse-dimension partitioning;
- Matmul/MMAD utilization;
- sparse tile sizing;
- UB/L1 buffering and pipeline overlap;
- online softmax / reduced intermediate traffic.

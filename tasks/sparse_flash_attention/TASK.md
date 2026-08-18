# SparseFlashAttention task contract

Status: scaffold. Fill this from the authoritative competition statement before implementation.

## Objective

Implement the competition-provided SparseFlashAttention operator in Ascend C, preserving the exact platform-visible interface and required semantics, then optimize scored performance cases without breaking hidden correctness cases.

## Authoritative inputs

Place or reference the official problem statement and submission template here. Do not silently substitute a generic FlashAttention API for the platform contract.

## Correctness contract

Document directly from the official statement:

- exact inputs / outputs / attributes and ordering;
- dtype and shape domain;
- sparse-index interpretation;
- causal/mask modes;
- RoPE/MLA score semantics;
- actual sequence-length semantics;
- optional outputs and numerical tolerances;
- invalid/padded sparse-index behavior.

## Performance objective

Document the platform scoring metric and target shapes when known. Visible scoring shapes do not remove hidden-test obligations unless the official statement says so.

## Allowed implementation changes

Internal Host Tiling, tiling data, workspace, kernel templates, memory planning, gather, Matmul/MMAD, softmax, multicore split, and pipeline strategy may be changed when the platform contract permits them.

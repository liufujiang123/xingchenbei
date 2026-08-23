# MhcExpand task contract

Status: scaffold.

The official CANNJudge template has been imported under `workspace/code/` from `MhcExpand_problem_301_template (1).zip`.

Before implementation, add the authoritative competition statement for this task and update this file from platform evidence. Do not infer missing semantics from template variable names.

## Objective

Implement and optimize the competition-provided `MhcExpand` Ascend C operator while preserving the exact platform-visible interface and required semantics.

## Authoritative inputs

- Official competition statement: not yet imported.
- Official template: `workspace/code/`.

## Correctness contract

TODO from the authoritative problem statement and platform metadata:

- exact inputs / outputs / attributes and ordering;
- dtype / shape / range domain;
- mathematical semantics;
- boundary and invalid-input behavior;
- output shape and dtype inference;
- numerical tolerance / reference behavior.

## Performance objective

TODO from the authoritative competition scoring rules.

## Interface rule

Treat the official template interface as immutable unless the competition platform explicitly states otherwise. Internal Host Tiling, tiling data, workspace, kernel templates, memory planning, multicore split, and pipeline strategy may be changed when the platform contract permits them.

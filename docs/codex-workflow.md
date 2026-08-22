# Codex workflow

Open the repository root as the VS Code workspace (or launch Codex from the repository root in CLI environments).

Suggested first instruction:

```text
Follow AGENTS.md. Inspect the task contract, problem statement, and official template.
Use $cannjudge-submit only for CANNJudge platform facts/actions, and use the installed AscendC skills for design, tiling, kernel, compile/debug, precision, and performance work.
Never request a plaintext CANNJudge password, never expose credential/key material, and do not submit unless I explicitly authorize submission.
First establish the simplest correct baseline. Do not optimize before build and correctness pass.
After baseline, use $xingchen-kernel-optimizer and docs/ascend-optimization-playbook.md.
Before changing performance code, classify the hot path as vector/cube/mixed_cv, draw the Ascend resource/dependency graph (MTE/V/Cube/Scalar/on-chip memory/workspace), identify the measured bottleneck, then generate a candidate shortlist with tools/ascend_perf_plan.py.
Treat pipeline overlap as a first-class optimization: MTE<->V for vector kernels and C(tile n+1)||V(tile n) for mixed Cube/Vector kernels when dependencies permit. Do not force C/V techniques onto pure Vector operators.
Run the evaluation-driven loop with one major hypothesis per candidate: guard, target build, correctness, same-case benchmark, profile when needed, then keep/reject with evidence.
```

Never infer success from source inspection. The configured shell commands and, when explicitly used, CANNJudge-returned submission evidence are the source of truth.

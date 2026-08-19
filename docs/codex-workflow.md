# Codex workflow

Open the repository root as the VS Code workspace (or launch Codex from the repository root in CLI environments).

Suggested first instruction:

```text
Follow AGENTS.md. Inspect the task contract, problem statement, and official template.
Use $cannjudge-submit only for CANNJudge platform facts/actions, and use the installed AscendC skills for design, tiling, kernel, compile/debug, precision, and performance work.
Never request a plaintext CANNJudge password, never expose credential/key material, and do not submit unless I explicitly authorize submission.
First establish the simplest correct baseline. Do not optimize before build and correctness pass.
After baseline, enter the evaluation-driven optimization loop: one hypothesis, one major change,
guard, build, correctness, benchmark, profile when needed, then keep/reject with evidence.
```

Never infer success from source inspection. The configured shell commands and, when explicitly used, CANNJudge-returned submission evidence are the source of truth.

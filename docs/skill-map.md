# Skill map

The repository activates a focused subset of the official `Ascend/agent-skills` repository, the official CANNJudge platform skill from `CANN/cann-learning-hub`, its supporting project skill, plus one local orchestration skill.

| Need | Skill |
|---|---|
| CANNJudge problem/package/submission/result/ranking interaction | `cannjudge-submit` |
| CANNJudge supporting operator-project reference | `ascendc-ops-project` |
| operator architecture, tiling/dataflow design | `ascendc-operator-design` |
| Ascend C Host/Kernel implementation | `ascendc-operator-code-gen` |
| Ascend C review | `ascendc-operator-code-review` |
| compiler/build diagnosis | `ascendc-operator-compile-debug` |
| memory/access sanitizer workflow | `ascendc-operator-mssanitizer` |
| numerical diagnosis | `ascendc-operator-precision-debug` |
| performance measurement/analysis | `ascendc-operator-performance-eval` |
| optimization implementation guidance | `ascendc-operator-performance-optim` |
| competition-safe eval loop | `xingchen-kernel-optimizer` |

## Routing rule

Use `cannjudge-submit` for platform facts and platform actions. Use the dedicated `Ascend/agent-skills` skills for core Ascend C technical decisions. `ascendc-ops-project` is installed because the official CANNJudge skill declares it as a dependency/supporting workflow, but it does not override repository policy or the more specialized Ascend skills.

CANNJudge submission is an external side effect: only perform it when the user explicitly authorizes submission. Never request or store plaintext passwords; follow the RSA-encrypted workflow documented by the official skill.

MIT KDA is kept as methodology/reference in `.agent-deps/kernel-design-agents`; its NVIDIA-specific skills are deliberately not installed into Codex's active repository skill set.

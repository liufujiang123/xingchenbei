# Skill map

The repository activates a focused subset of the official `Ascend/agent-skills` repository plus one local orchestration skill.

| Need | Skill |
|---|---|
| operator architecture, tiling/dataflow design | `ascendc-operator-design` |
| Ascend C Host/Kernel implementation | `ascendc-operator-code-gen` |
| Ascend C review | `ascendc-operator-code-review` |
| compiler/build diagnosis | `ascendc-operator-compile-debug` |
| memory/access sanitizer workflow | `ascendc-operator-mssanitizer` |
| numerical diagnosis | `ascendc-operator-precision-debug` |
| performance measurement/analysis | `ascendc-operator-performance-eval` |
| optimization implementation guidance | `ascendc-operator-performance-optim` |
| competition-safe eval loop | `xingchen-kernel-optimizer` |

MIT KDA is kept as methodology/reference in `.agent-deps/kernel-design-agents`; its NVIDIA-specific skills are deliberately not installed into Codex's active repository skill set.

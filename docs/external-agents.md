# External agent sources

This harness uses external projects by reference/bootstrap instead of vendoring mutable copies.

- `Ascend/agent-skills`: official Ascend domain skills. Selected AscendC skill directories are symlinked into `.agents/skills/`.
- `mit-han-lab/kernel-design-agents`: methodology/reference for candidate-driven kernel optimization. NVIDIA-specific skills are not installed as Ascend expertise.

Run `scripts/bootstrap_skills.sh` to clone/update dependencies into `.agent-deps/`.

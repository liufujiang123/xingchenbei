# External agent sources

This harness uses external projects by reference/bootstrap instead of vendoring mutable copies.

- `Ascend/agent-skills`: official Ascend domain skills. Selected AscendC skill directories are symlinked into `.agents/skills/`.
- `CANN/cann-learning-hub`: official CANN learning/competition skills. `cannjudge-submit` and its supporting `ascendc-ops-project` skill are symlinked into `.agents/skills/`.
- `mit-han-lab/kernel-design-agents`: methodology/reference for candidate-driven kernel optimization. NVIDIA-specific skills are not installed as Ascend expertise.

## Responsibility split

- CANNJudge platform truth/actions → `cannjudge-submit`
- Ascend C design/code/debug/precision/performance → `Ascend/agent-skills`
- Experiment discipline and candidate iteration → local `xingchen-kernel-optimizer` + MIT KDA methodology

The CANNJudge skill's authenticated mode uses RSA-encrypted credentials. Keep its key material and credentials local and untracked; never place plaintext passwords, private keys, tokens, or cookies in the repository.

Run `scripts/bootstrap_skills.sh` to clone/update dependencies into `.agent-deps/`.

# Xingchenbei Ascend Kernel Agent Harness

Codex-oriented harness for developing and optimizing Ascend C competition kernels.

Core layers:

1. Competition contract
2. Repository guardrails via `AGENTS.md`
3. Ascend expertise via official Agent Skills
4. Evaluation-driven optimization loop

Promotion order:

```text
build -> correctness -> benchmark -> optional profile -> promote/reject
```

The first target is SparseFlashAttention.

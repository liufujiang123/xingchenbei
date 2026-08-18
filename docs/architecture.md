# Architecture

The harness deliberately separates policy, expertise, and experiment control.

```text
Competition contract
       ↓
Repository AGENTS.md       ← immutable-interface / hidden-test guardrails
       ↓
Codex
  ↙          ↘
Ascend Skills   KDA-style loop
(domain APIs)   (hypothesis/eval discipline)
  ↘          ↙
CANN build + correctness evaluator + benchmark + profiler
       ↓
measured evidence
       ↓
next candidate
```

`AGENTS.md` is policy, not an Ascend textbook. Domain knowledge belongs in skills. Runtime commands belong in `config/agent.env`. Experiment results belong in `runs/` and the task optimization log.

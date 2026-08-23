# AGENTS.md

This repository is an AI coding-agent harness for Ascend C competition operators.
A closer nested `AGENTS.md`, when present, overrides this file.

## Red lines

- Preserve unrelated user work. Start with `git status --short`; do not revert or overwrite changes you do not own.
- Competition statement/platform contract and the official template outrank repository guidance, skills, references, and assumptions.
- Do not change the platform-visible operator contract unless authoritative competition evidence explicitly permits it.
- Do not guess hidden tests, narrow required functionality, weaken tolerances, or alter references to manufacture a pass.
- Build/correctness/performance claims require executed evidence. Local proxy results are not target-platform proof.
- External submission requires explicit user authorization. Never expose credentials, tokens, cookies, private keys, or credential ciphertext.

Implementation choices such as shape/SOC dispatch, tile/core/workspace planning, pipeline depth, and specialization belong in Host Tiling, tiling data, workspace, or internal templates—not new public inputs.

## Context economy

Treat the current model context as a cache.

At the start of a new Codex conversation for a task, run exactly once:

```bash
python3 tools/context_state.py bootstrap --task <task> --new-session
```

After that:

- unchanged file -> reuse current context; do not reread it;
- changed file -> inspect the presented diff first;
- diff insufficient -> read only the relevant section;
- full reread -> last resort;
- if context was compacted/lost, use `context_state.py use --force ...` for the specific file you need.

For an official or repository skill, load it through:

```bash
python3 tools/context_state.py use --task <task> <skill-path>
```

Do not repeatedly `cat`/`sed` stable `AGENTS.md`, Skill files, `TASK.md`, or `design.md` for orientation.

Do not narrate routine compliance (“I will first read...”, “I will follow the workflow...”); report only new facts, blockers, decisions, and results.

## Harness scope

Use the Harness for high-value mechanical work:

- context/version tracking;
- interface and task-scope protection;
- build -> correctness -> benchmark/profile evidence gates;
- compact task/run state;
- CANNJudge identity/package/result checks when configured;
- design/performance experience lookup when available.

Trust Codex for ordinary code reading, implementation, debugging, and general engineering judgment. Do not turn repository guidance into a checklist unless a concrete risk requires it.

Design experience is a catalog: inspect summaries once, choose relevant entries yourself, and load details only on demand. Machine suggestions are hints, not decisions.

After correctness is established, performance work should change one major mechanism at a time and keep/reject it from same-case evidence.

## Branch discipline

`main` is the canonical generic Harness baseline. Generic tools, guards, Skills, registries, and Harness documentation should land on `main` (or a dedicated infrastructure branch merged into `main`) and then reach active task branches through a merge from `main`.

Do not copy the same generic Harness fix independently into multiple task branches. Task branches should differ from `main` only for task-specific contract/code/config/tests/scripts/docs, plus explicit merges that inherit the canonical Harness. If task work reveals a reusable Ascend lesson, distill the reusable knowledge into the generic Harness deliberately rather than leaving infrastructure drift inside that task branch.

## Finish

Before finishing, verify intended diff scope, interface compatibility, executed validation evidence, and absence of generated artifacts or secrets.

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "repo=$ROOT"
for bin in git python3 bash rg; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "ok: $bin=$(command -v "$bin")"
  else
    echo "missing: $bin"
  fi
done
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
else
  echo "missing: npu-smi (expected off Ascend host)"
fi
if [[ -f "$ROOT/config/agent.env" ]]; then
  echo "ok: config/agent.env"
else
  echo "missing: config/agent.env"
fi
if [[ -f "$ROOT/.agents/skills/xingchen-kernel-optimizer/SKILL.md" ]]; then
  echo "ok: local optimizer skill"
else
  echo "missing: local optimizer skill"
fi
if [[ -d "$ROOT/.agents/skills/ascendc-operator-design" ]]; then
  echo "ok: Ascend skills bootstrapped"
else
  echo "missing: Ascend skills; run scripts/bootstrap_skills.sh"
fi

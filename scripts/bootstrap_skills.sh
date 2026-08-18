#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$ROOT/.agent-deps"
SKILLS="$ROOT/.agents/skills"
mkdir -p "$DEPS" "$SKILLS"

sync_repo() {
  local url="$1" dir="$2" branch="$3"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" fetch --depth=1 origin "$branch"
    git -C "$dir" reset --hard "origin/$branch"
  else
    git clone --depth=1 --branch "$branch" "$url" "$dir"
  fi
}

sync_repo https://github.com/Ascend/agent-skills.git "$DEPS/ascend-agent-skills" master
sync_repo https://github.com/mit-han-lab/kernel-design-agents.git "$DEPS/kernel-design-agents" main

ASCEND_BASE="$DEPS/ascend-agent-skills/skills"
for skill in \
  ascendc-operator-design \
  ascendc-operator-code-gen \
  ascendc-operator-code-review \
  ascendc-operator-compile-debug \
  ascendc-operator-mssanitizer \
  ascendc-operator-precision-debug \
  ascendc-operator-performance-eval \
  ascendc-operator-performance-optim; do
  src="$ASCEND_BASE/$skill"
  if [[ ! -f "$src/SKILL.md" ]]; then
    echo "Expected official skill not found: $src" >&2
    exit 1
  fi
  ln -sfn "$src" "$SKILLS/$skill"
done

# KDA is methodology/reference only. NVIDIA-specific skills are deliberately inactive.
ln -sfn "$DEPS/kernel-design-agents/prompts" "$ROOT/docs/kda-prompts"

echo "Installed repository-local skills:"
find "$SKILLS" -maxdepth 2 -name SKILL.md -print | sort

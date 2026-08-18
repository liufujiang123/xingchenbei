#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$ROOT/config/agent.env"

if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG. Copy config/agent.env.example and configure real commands." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CFG"

if [[ -n "${ASCEND_ENV_SETUP:-}" && -f "$ASCEND_ENV_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_ENV_SETUP"
fi

WORKSPACE="${WORKSPACE_DIR:-.}"
if [[ "$WORKSPACE" != /* ]]; then
  WORKSPACE="$ROOT/$WORKSPACE"
fi
if [[ ! -d "$WORKSPACE" ]]; then
  echo "Workspace does not exist: $WORKSPACE" >&2
  exit 2
fi

run_configured() {
  local name="$1"
  local cmd="${!name:-}"
  if [[ -z "$cmd" ]]; then
    echo "$name is empty in config/agent.env" >&2
    exit 2
  fi
  echo "cwd=$WORKSPACE" >&2
  echo "+ $cmd" >&2
  (cd "$WORKSPACE" && bash -lc "$cmd")
}

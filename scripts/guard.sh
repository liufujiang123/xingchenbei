#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "$ROOT/config/protected_paths.txt" && -f "$ROOT/config/protected_paths.sha256.json" ]]; then
  python3 "$ROOT/tools/interface_guard.py" check
else
  echo "interface guard not configured; skipping"
fi

# Evidence freshness is fail-open: inability to fingerprint must not block kernel
# work, but then task_state will report old evidence as UNKNOWN rather than fresh.
if [[ -f "$ROOT/tools/evidence_fingerprint.py" ]]; then
  TASK="${TASK_NAME:-}"
  if [[ -z "$TASK" && -f "$ROOT/config/agent.env" ]]; then
    TASK="$(
      bash -lc "set -a; source '$ROOT/config/agent.env'; printf '%s' \"\${TASK_NAME:-}\"" \
        2>/dev/null || true
    )"
  fi
  if [[ -n "$TASK" ]]; then
    python3 "$ROOT/tools/evidence_fingerprint.py" \
      --task "$TASK" \
      --prepare-best \
      --emit-marker \
      || echo "HARNESS_EVIDENCE_FINGERPRINT=UNKNOWN"
  fi
fi

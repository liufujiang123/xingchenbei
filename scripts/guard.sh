#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/config/protected_paths.txt" && -f "$ROOT/config/protected_paths.sha256.json" ]]; then
  python3 "$ROOT/tools/interface_guard.py" check
else
  echo "interface guard not configured; skipping"
fi

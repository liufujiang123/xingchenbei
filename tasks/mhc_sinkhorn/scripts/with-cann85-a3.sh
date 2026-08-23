#!/usr/bin/env bash
set -euo pipefail

[[ $# -gt 0 ]] || { echo "Usage: $0 <command> [args...]" >&2; exit 2; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CANN85_INSTALL_ROOT="${CANN85_A3_INSTALL_ROOT:-${HOME}/ascend-envs/cann-8.5.0-a3}"
export CANN85_EXPECTED_OPS_PACKAGE=Ascend-cann-A3-ops
exec "${script_dir}/with-cann85.sh" "$@"

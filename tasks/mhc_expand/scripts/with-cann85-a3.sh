#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
install_root="${CANN85_A3_INSTALL_ROOT:-${HOME}/ascend-envs/cann-8.5.0-a3}"

export CANN85_INSTALL_ROOT="${install_root}"
export CANN85_EXPECTED_OPS_PACKAGE="Ascend-cann-A3-ops"
exec "${script_dir}/with-cann85.sh" "$@"

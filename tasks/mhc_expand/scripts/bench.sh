#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
label=${MHC_EXPAND_BENCH_LABEL:-harness_$(date -u +%Y%m%dt%H%M%Sz)}

exec "${script_dir}/bench-local-a3.sh" "${label}"

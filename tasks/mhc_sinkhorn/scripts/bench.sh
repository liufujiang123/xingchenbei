#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
label=${MHC_SINKHORN_BENCH_LABEL:-harness_$(date -u +%Y%m%dt%H%M%Sz)}
exec "${script_dir}/bench-local-a3.sh" "${label}"

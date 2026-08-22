#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)

fail() {
    echo "LOCAL_A3_BENCH_FAIL: $*" >&2
    exit 1
}

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -eq 1 ]] || fail "usage: $0 <label>"
    label=$1
    [[ "${label}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail "invalid label: ${label}"
    device_id=${MHC_EXPAND_DEVICE_ID:-4}
    [[ "${device_id}" =~ ^[0-9]+$ && "${device_id}" != "0" ]] || \
        fail "MHC_EXPAND_DEVICE_ID must be a nonzero integer"
    export MHC_EXPAND_DEVICE_ID="${device_id}"
    exec "${script_dir}/with-cann85-a3.sh" \
        "${script_path}" --inside-a3-cann85 "${label}"
fi

shift
[[ $# -eq 1 ]] || fail "invalid internal invocation"
label=$1
managed_root="${TMPDIR:-/tmp}/mhc_expand_a3_validation"
install_dir="${managed_root}/install"
index_path="${install_dir}/vendors/custom/op_impl/ai_core/tbe/kernel/config/ascend910_93/binary_info_config.json"
custom_set_env="${install_dir}/vendors/custom/bin/set_env.bash"

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-A3-ops" ]] || fail "A3 OPS package is not active"
[[ -r "${index_path}" ]] || fail "missing local A3 package; run validate-local-a3.sh first"
grep -q 'MhcExpand' "${index_path}" || fail "installed index does not contain MhcExpand"
[[ -r "${custom_set_env}" ]] || fail "custom OPP environment script is missing"

set +u
# shellcheck disable=SC1090
source "${custom_set_env}"
set -u
export MHC_EXPAND_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"
output="${task_dir}/runs/performance/mhc_expand_${label}.md"

echo "LOCAL A3 PERFORMANCE PROXY"
echo "NOT CANNJUDGE 910B PERFORMANCE"
echo "LABEL=${label}"
echo "DEVICE_ID=${MHC_EXPAND_DEVICE_ID}"
python -u "${task_dir}/tests/benchmark_mhc_expand_acl.py" \
    --cases "${task_dir}/tests/mhc_expand_perf_cases.jsonl" \
    --output "${output}" \
    --label "${label}"

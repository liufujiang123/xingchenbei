#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
fail() { echo "LOCAL_A3_BENCH_FAIL: $*" >&2; exit 1; }

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -eq 1 ]] || fail "usage: $0 <label>"
    label=$1
    [[ "${label}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail "invalid label: ${label}"
    device_id=${MHC_SINKHORN_DEVICE_ID:-4}
    [[ "${device_id}" =~ ^[0-9]+$ && "${device_id}" != "0" ]] || fail "use a valid nonzero device"
    export MHC_SINKHORN_DEVICE_ID="${device_id}"
    exec "${script_dir}/with-cann85-a3.sh" "${script_path}" --inside-a3-cann85 "${label}"
fi
shift
[[ $# -eq 1 ]] || fail "invalid internal invocation"
label=$1
managed_root="${TMPDIR:-/tmp}/mhc_sinkhorn_a3_validation"
install_dir="${managed_root}/install"
index_path="${install_dir}/vendors/custom/op_impl/ai_core/tbe/kernel/config/ascend910_93/binary_info_config.json"
custom_set_env="${install_dir}/vendors/custom/bin/set_env.bash"
source_digest_path="${managed_root}/logs/source.sha256"

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-A3-ops" ]] || fail "A3 OPS package is not active"
[[ -r "${index_path}" ]] || fail "run validate-local-a3.sh before benchmarking"
grep -q 'MhcSinkhorn' "${index_path}" || fail "installed index does not contain MhcSinkhorn"
[[ -r "${custom_set_env}" && -r "${source_digest_path}" ]] || fail "validated source manifest is missing"
current_digest=$(find "${task_dir}/workspace/code" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d " " -f 1)
validated_digest=$(sed -n 's/^SOURCE_SHA256=//p' "${source_digest_path}" | head -1)
[[ "${current_digest}" == "${validated_digest}" ]] || fail "installed A3 package is stale; rerun validate-local-a3.sh"

set +u
# shellcheck disable=SC1090
source "${custom_set_env}"
set -u
export MHC_SINKHORN_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"
output_dir="${task_dir}/runs/performance"
output_jsonl="${output_dir}/mhc_sinkhorn_${label}.jsonl"
output_report="${output_dir}/mhc_sinkhorn_${label}.md"

echo "LOCAL A3 MHC-SINKHORN PERFORMANCE"
echo "EVIDENCE_CLASS=benchmark_observed"
echo "NOT CANNJUDGE 910B PERFORMANCE"
echo "LABEL=${label}"
echo "DEVICE_ID=${MHC_SINKHORN_DEVICE_ID}"
python -u "${task_dir}/tests/benchmark_mhc_sinkhorn_acl.py" \
    --cases "${task_dir}/tests/mhc_sinkhorn_perf_cases.jsonl" \
    --output-jsonl "${output_jsonl}" \
    --output-report "${output_report}" \
    --label "${label}"

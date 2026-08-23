#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
fail() { echo "LOCAL_A3_PROFILE_FAIL: $*" >&2; exit 1; }

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -ge 1 && $# -le 2 ]] || fail "usage: $0 <label> [case-id]"
    label=$1
    case_id=${2:-mc_192}
    [[ "${label}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail "invalid label: ${label}"
    [[ "${case_id}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail "invalid case id: ${case_id}"
    device_id=${MHC_SINKHORN_DEVICE_ID:-4}
    [[ "${device_id}" =~ ^[0-9]+$ && "${device_id}" != "0" ]] || fail "use a valid nonzero device"
    export MHC_SINKHORN_DEVICE_ID="${device_id}"
    exec "${script_dir}/with-cann85-a3.sh" "${script_path}" --inside-a3-cann85 "${label}" "${case_id}"
fi
shift
[[ $# -eq 2 ]] || fail "invalid internal invocation"
label=$1
case_id=$2
managed_root="${TMPDIR:-/tmp}/mhc_sinkhorn_a3_validation"
install_dir="${managed_root}/install"
custom_set_env="${install_dir}/vendors/custom/bin/set_env.bash"
source_digest_path="${managed_root}/logs/source.sha256"
profile_root="${TMPDIR:-/tmp}/mhc_sinkhorn_profile/${label}"
aic_metrics=${MHC_SINKHORN_AIC_METRICS:-PipeUtilization}
case "${aic_metrics}" in
    PipeUtilization|ArithmeticUtilization|Memory|MemoryL0|ResourceConflictRatio|MemoryUB|L2Cache|MemoryAccess) ;;
    *) fail "unsupported MHC_SINKHORN_AIC_METRICS: ${aic_metrics}" ;;
esac

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
[[ -r "${custom_set_env}" && -r "${source_digest_path}" ]] || fail "run validate-local-a3.sh first"
current_digest=$(find "${task_dir}/workspace/code" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d " " -f 1)
validated_digest=$(sed -n 's/^SOURCE_SHA256=//p' "${source_digest_path}" | head -1)
[[ "${current_digest}" == "${validated_digest}" ]] || fail "installed A3 package is stale; rerun validate-local-a3.sh"

set +u
# shellcheck disable=SC1090
source "${custom_set_env}"
set -u
export MHC_SINKHORN_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"
[[ -r "${MHC_SINKHORN_CUSTOM_LIB}" ]] || fail "custom ACLNN library is missing"
command -v msprof >/dev/null || { echo "PROFILE_UNAVAILABLE=msprof_not_found"; exit 0; }

if [[ -d "${profile_root}" ]]; then
    find "${profile_root}" -type d -exec chmod u+rwx {} +
    find "${profile_root}" -type f -exec chmod u+rw {} +
    find "${profile_root}" -depth -delete
elif [[ -e "${profile_root}" ]]; then
    fail "profile root exists but is not a directory"
fi
mkdir -p "${profile_root}"
chmod 700 "${profile_root}"
workload="python3 -u ${task_dir}/tests/benchmark_mhc_sinkhorn_acl.py --cases ${task_dir}/tests/mhc_sinkhorn_perf_cases.jsonl --output-jsonl ${profile_root}/workload.jsonl --output-report ${profile_root}/workload.md --label ${label} --case-id ${case_id}"

echo "LOCAL A3 MSPROF ATTEMPT"
echo "EVIDENCE_CLASS=profile_observed_if_export_succeeds"
echo "CASE=${case_id}"
echo "AIC_METRICS=${aic_metrics}"
echo "RAW_PROFILE_ROOT=${profile_root}"
set +e
timeout "${MHC_SINKHORN_PROFILE_TIMEOUT:-240}" msprof \
    --application="${workload}" \
    --output="${profile_root}" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=on \
    --ai-core=on \
    --aic-mode=task-based \
    --aic-metrics="${aic_metrics}" \
    --sys-hardware-mem=off \
    --type=text \
    >"${profile_root}/msprof.log" 2>&1
profile_rc=$?
set -e
tail -n 80 "${profile_root}/msprof.log"
if [[ ${profile_rc} -ne 0 ]]; then
    echo "PROFILE_UNAVAILABLE=msprof_exit_${profile_rc}"
    exit 0
fi

csv_count=$(find "${profile_root}" -type f -name '*.csv' | wc -l)
if [[ ${csv_count} -eq 0 ]]; then
    echo "PROFILE_UNAVAILABLE=no_exported_csv"
    exit 0
fi
echo "PROFILE_OBSERVED=1"
echo "PROFILE_CSV_COUNT=${csv_count}"
find "${profile_root}" -type f -name '*.csv' -print | sort
profile_report="${task_dir}/runs/performance/mhc_sinkhorn_profile_${label}.md"
python3 "${task_dir}/tests/summarize_mhc_sinkhorn_profile.py" \
    --profile-root "${profile_root}" \
    --output "${profile_report}" \
    --label "${label}"
echo "PROFILE_REPORT=${profile_report}"

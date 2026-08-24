#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
fixture="${task_dir}/tests/fixtures/sparse_flash_attention_minimal_host.cpp"
test_file="${task_dir}/tests/test_sparse_flash_attention.py"
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}

fail() { echo "SFA_910B_TILING_CONTROL_FAIL: $*" >&2; exit 1; }
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || fail "ASCEND_HOME_PATH is invalid"
[[ -r "${fixture}" ]] || fail "missing minimal Host fixture"

run_root=${SFA_910B_TILING_CONTROL_ROOT:-"${TMPDIR:-/tmp}/sfa-910b-tiling-control"}
if [[ -e "${run_root}" ]]; then
    [[ -d "${run_root}" ]] || fail "control root is not a directory"
    find "${run_root}" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "${run_root}" -type f -exec chmod u+rw {} + 2>/dev/null || true
    find "${run_root}" -depth -delete
fi
mkdir -p "${run_root}/variants" "${run_root}/logs" "${run_root}/results"
printf 'variant\tbuild\tget_workspace\n' >"${run_root}/results/summary.tsv"

run_variant() {
    local variant=$1
    local variant_root="${run_root}/variants/${variant}"
    local variant_source="${variant_root}/source"
    local build_dir="${variant_root}/build"
    mkdir -p "${variant_source}" "${build_dir}"
    cp -a "${source_dir}/." "${variant_source}/"
    if [[ "${variant}" == "minimal_host_control" ]]; then
        cp "${fixture}" "${variant_source}/op_host/sparse_flash_attention.cpp"
    fi
    local build_log="${run_root}/logs/${variant}-build.log"
    echo "TILING_CONTROL_VARIANT=${variant}"
    if ! cmake -S "${variant_source}" -B "${build_dir}" \
            -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
            >"${build_log}" 2>&1 || \
       ! cmake --build "${build_dir}" --parallel "${SFA_910B_BUILD_JOBS:-2}" \
            >>"${build_log}" 2>&1; then
        printf '%s\tFAIL\t-\n' "${variant}" >>"${run_root}/results/summary.tsv"
        return
    fi
    local custom_lib="${build_dir}/libcust_opapi.so"
    local tiling_lib="${build_dir}/op_host/libcustom_ascendc_cust_optiling.so"
    if [[ ! -r "${custom_lib}" || ! -r "${tiling_lib}" ]]; then
        printf '%s\tPASS\tMISSING_LIBRARY\n' "${variant}" >>"${run_root}/results/summary.tsv"
        return
    fi
    local case_log="${run_root}/logs/${variant}-workspace.log"
    if SPARSE_FLASH_ATTENTION_CUSTOM_LIB="${custom_lib}" \
       SPARSE_FLASH_ATTENTION_TILING_LIB="${tiling_lib}" \
       timeout "${SFA_910B_CASE_TIMEOUT:-180}" python3 -u "${test_file}" \
            D_rope_required --workspace-only --device "${device_id}" >"${case_log}" 2>&1; then
        printf '%s\tPASS\tPASS\n' "${variant}" >>"${run_root}/results/summary.tsv"
    else
        printf '%s\tPASS\tFAIL\n' "${variant}" >>"${run_root}/results/summary.tsv"
    fi
}

run_variant minimal_host_control
run_variant production_host

archive="${TMPDIR:-/tmp}/sfa-910b-tiling-control-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "${archive}" -C "${run_root}" results logs
echo "===== SFA 910B TILING CONTROL SUMMARY ====="
column -t -s $'\t' "${run_root}/results/summary.tsv" 2>/dev/null || cat "${run_root}/results/summary.tsv"
echo "RESULT_ARCHIVE=${archive}"
if [[ "${SFA_910B_PUSH_RESULTS:-0}" == "1" ]]; then
    "${script_dir}/upload-910b-probe-results.sh" "${archive}"
fi

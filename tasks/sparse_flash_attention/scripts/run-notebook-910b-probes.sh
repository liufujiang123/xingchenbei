#!/usr/bin/env bash
set -uo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
test_file="${task_dir}/tests/test_sparse_flash_attention.py"
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}

fail() {
    echo "SFA_910B_PROBE_FAIL: $*" >&2
    exit 1
}

[[ "${device_id}" =~ ^[0-9]+$ ]] || fail "invalid device id: ${device_id}"
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || \
    fail "activate CANN first; ASCEND_HOME_PATH is missing or invalid"
[[ -r "${test_file}" ]] || fail "missing runner: ${test_file}"
command -v cmake >/dev/null || fail "cmake is unavailable"
command -v python3 >/dev/null || fail "python3 is unavailable"
command -v npu-smi >/dev/null || fail "npu-smi is unavailable"

mapfile -t toolkit_infos < <(find "${ASCEND_HOME_PATH}" -maxdepth 4 \
    -type f -name ascend_toolkit_install.info -print 2>/dev/null)
[[ ${#toolkit_infos[@]} -ge 1 ]] || fail "CANN toolkit metadata not found below ASCEND_HOME_PATH"
version_ok=0
for toolkit_info in "${toolkit_infos[@]}"; do
    if grep -qx 'version=8.5.0' "${toolkit_info}"; then
        version_ok=1
        break
    fi
done
[[ ${version_ok} -eq 1 ]] || fail "active toolkit is not verified as CANN 8.5.0"

run_root=${SFA_910B_PROBE_ROOT:-"${TMPDIR:-/tmp}/sfa-910b-probes"}
if [[ -e "${run_root}" ]]; then
    [[ -d "${run_root}" ]] || fail "probe root is not a directory: ${run_root}"
    find "${run_root}" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "${run_root}" -type f -exec chmod u+rw {} + 2>/dev/null || true
    find "${run_root}" -depth -delete
fi
mkdir -p "${run_root}/variants" "${run_root}/logs" "${run_root}/results"

repo_commit=$(git -C "${task_dir}" rev-parse HEAD 2>/dev/null || echo unknown)
{
    echo "UTC_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "REPO_COMMIT=${repo_commit}"
    echo "ASCEND_HOME_PATH=$(readlink -f "${ASCEND_HOME_PATH}")"
    echo "DEVICE_ID=${device_id}"
    echo "PYTHON=$(python3 -V 2>&1)"
    echo "CMAKE=$(cmake --version | head -1)"
    npu-smi info
} >"${run_root}/results/environment.txt" 2>&1

variants=(baseline rope_scalar_ub exp_scalar rope_scalar_ub_exp_scalar)
cases=(D_rope_required D_rope_required_fp32 D_rope_required_bf16 A_basic)
printf 'variant\tbuild\tcase\trun\n' >"${run_root}/results/summary.tsv"

for variant in "${variants[@]}"; do
    variant_root="${run_root}/variants/${variant}"
    variant_source="${variant_root}/source"
    build_dir="${variant_root}/build"
    mkdir -p "${variant_source}" "${build_dir}"
    cp -a "${source_dir}/." "${variant_source}/"

    case "${variant}" in
        baseline) ;;
        rope_scalar_ub)
            sed -i \
                's/ROPE_DOT_EXPERIMENT = RopeDotExperiment::VECTOR_REDUCE/ROPE_DOT_EXPERIMENT = RopeDotExperiment::SCALAR_UB/' \
                "${variant_source}/op_kernel/sparse_flash_attention.cpp"
            ;;
        exp_scalar)
            sed -i \
                's/EXP_EXPERIMENT = ExpExperiment::VECTOR/EXP_EXPERIMENT = ExpExperiment::SCALAR_POLYNOMIAL/' \
                "${variant_source}/op_kernel/sparse_flash_attention.cpp"
            ;;
        rope_scalar_ub_exp_scalar)
            sed -i \
                -e 's/ROPE_DOT_EXPERIMENT = RopeDotExperiment::VECTOR_REDUCE/ROPE_DOT_EXPERIMENT = RopeDotExperiment::SCALAR_UB/' \
                -e 's/EXP_EXPERIMENT = ExpExperiment::VECTOR/EXP_EXPERIMENT = ExpExperiment::SCALAR_POLYNOMIAL/' \
                "${variant_source}/op_kernel/sparse_flash_attention.cpp"
            ;;
        *) fail "unknown variant: ${variant}" ;;
    esac

    build_log="${run_root}/logs/${variant}-build.log"
    if cmake -S "${variant_source}" -B "${build_dir}" \
            -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
            >"${build_log}" 2>&1 && \
       cmake --build "${build_dir}" --parallel "${SFA_910B_BUILD_JOBS:-2}" \
            >>"${build_log}" 2>&1; then
        build_status=PASS
    else
        build_status=FAIL
    fi
    printf '%s\t%s\t-\t-\n' "${variant}" "${build_status}" \
        >>"${run_root}/results/summary.tsv"
    if [[ "${build_status}" != PASS ]]; then
        continue
    fi

    custom_lib="${build_dir}/libcust_opapi.so"
    tiling_lib="${build_dir}/op_host/libcustom_ascendc_cust_optiling.so"
    if [[ ! -r "${custom_lib}" || ! -r "${tiling_lib}" ]]; then
        printf '%s\tPASS\t-\tMISSING_CUSTOM_OR_TILING_LIB\n' "${variant}" \
            >>"${run_root}/results/summary.tsv"
        continue
    fi
    export SPARSE_FLASH_ATTENTION_CUSTOM_LIB="${custom_lib}"
    export SPARSE_FLASH_ATTENTION_TILING_LIB="${tiling_lib}"
    export SFA_REFERENCE_AGGREGATION=value
    export SFA_REFERENCE_SCALE=attribute
    export SFA_REFERENCE_ROPE_SCALE=scaled
    export SFA_REFERENCE_CAUSAL=right_down_actual
    export SFA_REFERENCE_SPARSE_UNIT=block
    export SFA_REFERENCE_ROPE=enabled

    for case_name in "${cases[@]}"; do
        case_log="${run_root}/logs/${variant}-${case_name}.log"
        if timeout "${SFA_910B_CASE_TIMEOUT:-180}" \
            python3 -u "${test_file}" "${case_name}" --device "${device_id}" \
            >"${case_log}" 2>&1; then
            run_status=PASS
        else
            status=$?
            if [[ ${status} -eq 124 ]]; then
                run_status=TIMEOUT
            else
                run_status="FAIL_${status}"
            fi
        fi
        printf '%s\tPASS\t%s\t%s\n' "${variant}" "${case_name}" "${run_status}" \
            >>"${run_root}/results/summary.tsv"
        grep -E '^(\{|SUMMARY|CAUSAL_PREDICATE)' "${case_log}" \
            >"${run_root}/results/${variant}-${case_name}.txt" 2>/dev/null || true
    done
done

echo "UTC_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${run_root}/results/environment.txt"
archive="${TMPDIR:-/tmp}/sfa-910b-probe-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "${archive}" -C "${run_root}" results logs

echo ""
echo "===== SFA 910B PROBE SUMMARY ====="
column -t -s $'\t' "${run_root}/results/summary.tsv" 2>/dev/null || \
    cat "${run_root}/results/summary.tsv"
echo "RESULT_ARCHIVE=${archive}"
if [[ "${SFA_910B_PUSH_RESULTS:-0}" == "1" ]]; then
    "${script_dir}/upload-910b-probe-results.sh" "${archive}"
else
    echo "Upload that archive to the Codex chat; it contains no credentials."
    echo "Set SFA_910B_PUSH_RESULTS=1 after configuring GitHub write authentication to push it automatically."
fi

#!/usr/bin/env bash
# Isolated 910B experiments for UB positions, ReduceSum result extraction, and
# the RopeDot/Exp 2x2.  Every variant is packaged and installed independently;
# the competition workspace and public interface remain unchanged.
set -uo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
test_file="${task_dir}/tests/test_sparse_flash_attention.py"
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}

fail() { echo "SFA_910B_DEVICE_API_PROBE_FAIL: $*" >&2; exit 1; }
[[ "${device_id}" =~ ^[0-9]+$ ]] || fail "invalid device id: ${device_id}"
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || \
    fail "activate CANN first; ASCEND_HOME_PATH is missing or invalid"
[[ -r "${test_file}" ]] || fail "missing test runner"

run_root=${SFA_910B_DEVICE_API_PROBE_ROOT:-"${TMPDIR:-/tmp}/sfa-910b-device-api-probes"}
if [[ -e "${run_root}" ]]; then
    [[ -d "${run_root}" ]] || fail "probe root is not a directory"
    find "${run_root}" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "${run_root}" -type f -exec chmod u+rw {} + 2>/dev/null || true
    find "${run_root}" -depth -delete
fi
mkdir -p "${run_root}/variants" "${run_root}/logs" \
    "${run_root}/results/arrays"

variants=(
    baseline
    standard_positions
    getacc
    rope_scalar_exp_vector
    rope_vector_exp_scalar
    rope_scalar_exp_scalar
)
cases=(
    L_rope_single_index
    L_content_single_index
    D_rope_required
    D_rope_required_fp32
    A_basic
)
printf 'variant\tpackage\tinstall\tcase\trun\n' >"${run_root}/results/summary.tsv"

apply_variant() {
    local variant=$1
    local kernel=$2
    case "${variant}" in
        baseline) ;;
        standard_positions)
            sed -i \
                's/BufferPositionExperiment::LEGACY_VECCALC;/BufferPositionExperiment::STANDARD_VECIN_VECOUT;/' \
                "${kernel}"
            ;;
        getacc)
            sed -i \
                's/DotResultExperiment::LOCAL_TENSOR;/DotResultExperiment::ACCUMULATOR_REGISTER;/' \
                "${kernel}"
            ;;
        rope_scalar_exp_vector)
            sed -i \
                's/ROPE_DOT_EXPERIMENT = RopeDotExperiment::VECTOR_REDUCE/ROPE_DOT_EXPERIMENT = RopeDotExperiment::SCALAR_UB/' \
                "${kernel}"
            ;;
        rope_vector_exp_scalar)
            sed -i \
                's/EXP_EXPERIMENT = ExpExperiment::VECTOR/EXP_EXPERIMENT = ExpExperiment::SCALAR_POLYNOMIAL/' \
                "${kernel}"
            ;;
        rope_scalar_exp_scalar)
            sed -i \
                -e 's/ROPE_DOT_EXPERIMENT = RopeDotExperiment::VECTOR_REDUCE/ROPE_DOT_EXPERIMENT = RopeDotExperiment::SCALAR_UB/' \
                -e 's/EXP_EXPERIMENT = ExpExperiment::VECTOR/EXP_EXPERIMENT = ExpExperiment::SCALAR_POLYNOMIAL/' \
                "${kernel}"
            ;;
        *) fail "unknown variant: ${variant}" ;;
    esac
}

for variant in "${variants[@]}"; do
    echo "DEVICE_API_PROBE_VARIANT=${variant}"
    variant_root="${run_root}/variants/${variant}"
    variant_source="${variant_root}/source"
    build_dir="${variant_root}/build"
    install_dir="${variant_root}/install"
    mkdir -p "${variant_source}" "${build_dir}" "${install_dir}" \
        "${run_root}/results/arrays/${variant}"
    cp -a "${source_dir}/." "${variant_source}/"
    apply_variant "${variant}" \
        "${variant_source}/op_kernel/sparse_flash_attention.cpp"
    sed -i 's/TYPE SHARED/TYPE RUN/' "${variant_source}/CMakeLists.txt"

    build_log="${run_root}/logs/${variant}-build-package.log"
    if ! cmake -S "${variant_source}" -B "${build_dir}" \
            -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
            >"${build_log}" 2>&1 || \
       ! cmake --build "${build_dir}" --parallel "${SFA_910B_BUILD_JOBS:-2}" \
            >>"${build_log}" 2>&1 || \
       ! cmake --build "${build_dir}" --target SparseFlashAttention_ascend910b \
            --parallel "${SFA_910B_BUILD_JOBS:-2}" >>"${build_log}" 2>&1 || \
       ! cmake --build "${build_dir}" \
            --target ascendc_kernels_ascendc_bin_ascend910b_gen_ops_config \
            --parallel "${SFA_910B_BUILD_JOBS:-2}" >>"${build_log}" 2>&1 || \
       ! cpack --config "${build_dir}/CPackConfig.cmake" >>"${build_log}" 2>&1; then
        printf '%s\tFAIL\t-\t-\t-\n' "${variant}" \
            >>"${run_root}/results/summary.tsv"
        continue
    fi

    package_path=$(find "${build_dir}" -type f -name '*.run' -print -quit)
    install_log="${run_root}/logs/${variant}-install.log"
    if [[ -z "${package_path}" ]] || \
       ! env -u ASCEND_CUSTOM_OPP_PATH bash "${package_path}" \
            --install-path="${install_dir}" --quiet >"${install_log}" 2>&1; then
        printf '%s\tPASS\tFAIL\t-\t-\n' "${variant}" \
            >>"${run_root}/results/summary.tsv"
        continue
    fi
    set_env=$(find "${install_dir}/vendors" -path '*/bin/set_env.bash' \
        -type f -print -quit)
    custom_lib=$(find "${install_dir}/vendors" -path '*/op_api/lib/libcust_opapi.so' \
        -type f -print -quit)
    if [[ -z "${set_env}" || -z "${custom_lib}" ]]; then
        printf '%s\tPASS\tMISSING_ARTIFACT\t-\t-\n' "${variant}" \
            >>"${run_root}/results/summary.tsv"
        continue
    fi

    for case_name in "${cases[@]}"; do
        case_log="${run_root}/logs/${variant}-${case_name}.log"
        if env -u ASCEND_CUSTOM_OPP_PATH -u SPARSE_FLASH_ATTENTION_TILING_LIB \
               -u LD_PRELOAD bash -c '
                   source "$1"
                   SPARSE_FLASH_ATTENTION_CUSTOM_LIB="$2" \
                   timeout "$3" python3 -u "$4" "$5" --device "$6" \
                       --dump-output-dir "$7"
               ' bash "${set_env}" "${custom_lib}" \
               "${SFA_910B_CASE_TIMEOUT:-180}" "${test_file}" "${case_name}" \
               "${device_id}" "${run_root}/results/arrays/${variant}" \
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
        printf '%s\tPASS\tPASS\t%s\t%s\n' \
            "${variant}" "${case_name}" "${run_status}" \
            >>"${run_root}/results/summary.tsv"
    done
done

python3 - "${run_root}/results/arrays" >"${run_root}/results/differential.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
baseline = root / "baseline"
report = []
for candidate in sorted(path for path in root.iterdir() if path.name != "baseline"):
    for expected_path in sorted(baseline.glob("*.npy")):
        actual_path = candidate / expected_path.name
        if not actual_path.exists():
            report.append({"variant": candidate.name, "array": expected_path.name, "status": "MISSING"})
            continue
        expected = np.load(expected_path)
        actual = np.load(actual_path)
        expected_num = expected.astype(np.float64)
        actual_num = actual.astype(np.float64)
        report.append({
            "variant": candidate.name,
            "array": expected_path.name,
            "status": "OK",
            "dtype": str(actual.dtype),
            "bitwise_equal": bool(np.array_equal(actual.view(np.uint8), expected.view(np.uint8))),
            "max_abs": float(np.max(np.abs(actual_num - expected_num), initial=0.0)),
            "different_elements": int(np.count_nonzero(actual != expected)),
        })
print(json.dumps(report, sort_keys=True, indent=2))
PY

{
    echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "REPO_COMMIT=$(git -C "${task_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ASCEND_HOME_PATH=$(readlink -f "${ASCEND_HOME_PATH}")"
    echo "DEVICE_ID=${device_id}"
    npu-smi info
} >"${run_root}/results/environment.txt" 2>&1

archive="${TMPDIR:-/tmp}/sfa-910b-device-api-probes-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "${archive}" -C "${run_root}" results logs
echo "===== SFA 910B DEVICE API PROBE SUMMARY ====="
column -t -s $'\t' "${run_root}/results/summary.tsv" 2>/dev/null || \
    command cat "${run_root}/results/summary.tsv"
echo "RESULT_ARCHIVE=${archive}"
if [[ "${SFA_910B_PUSH_RESULTS:-0}" == "1" ]]; then
    "${script_dir}/upload-910b-probe-results.sh" "${archive}"
fi

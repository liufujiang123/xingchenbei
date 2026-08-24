#!/usr/bin/env bash
# Build a testing-only RUN package, install it in an isolated temporary root,
# and call ACLNN through the installed OPP.  This distinguishes a loose-.so
# registry/discovery failure from an operator Host/Tiling failure on 910B.
set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
fixture="${task_dir}/tests/fixtures/sparse_flash_attention_minimal_host.cpp"
test_file="${task_dir}/tests/test_sparse_flash_attention.py"
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}

fail() { echo "SFA_910B_OPP_INSTALL_CONTROL_FAIL: $*" >&2; exit 1; }
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || fail "ASCEND_HOME_PATH is invalid"
[[ -r "${fixture}" ]] || fail "missing minimal Host fixture"

run_root=${SFA_910B_OPP_INSTALL_CONTROL_ROOT:-"${TMPDIR:-/tmp}/sfa-910b-opp-install-control"}
if [[ -e "${run_root}" ]]; then
    [[ -d "${run_root}" ]] || fail "control root is not a directory"
    find "${run_root}" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "${run_root}" -type f -exec chmod u+rw {} + 2>/dev/null || true
    find "${run_root}" -depth -delete
fi
mkdir -p "${run_root}/source" "${run_root}/build" "${run_root}/install" \
    "${run_root}/logs" "${run_root}/results"

cp -a "${source_dir}/." "${run_root}/source/"
cp "${fixture}" "${run_root}/source/op_host/sparse_flash_attention.cpp"

# RUN is deliberately applied only to this copied test project.  The official
# FP16/FP32 support list removes the unrelated BF16 extension from this test.
sed -i \
    -e 's/TYPE SHARED/TYPE RUN/' \
    "${run_root}/source/CMakeLists.txt"
sed -i \
    -e 's/{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}/{ge::DT_FLOAT16, ge::DT_FLOAT}/g' \
    -e 's/{ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}/{ge::DT_INT32, ge::DT_INT32}/g' \
    -e 's/{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}/{ge::DT_FLOAT, ge::DT_FLOAT}/g' \
    -e 's/{ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}/{ge::FORMAT_ND, ge::FORMAT_ND}/g' \
    "${run_root}/source/op_host/sparse_flash_attention.cpp"

build_log="${run_root}/logs/build-and-package.log"
# CANN 8.5's RUN package does not make the per-SOC binary target a default
# dependency. CPack requires its output directory, so build it explicitly.
if ! cmake -S "${run_root}/source" -B "${run_root}/build" \
        -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
        >"${build_log}" 2>&1 || \
   ! cmake --build "${run_root}/build" --parallel "${SFA_910B_BUILD_JOBS:-2}" \
        >>"${build_log}" 2>&1 || \
   ! cmake --build "${run_root}/build" --target SparseFlashAttention_ascend910b \
        --parallel "${SFA_910B_BUILD_JOBS:-2}" >>"${build_log}" 2>&1 || \
   ! cmake --build "${run_root}/build" \
        --target ascendc_kernels_ascendc_bin_ascend910b_gen_ops_config \
        --parallel "${SFA_910B_BUILD_JOBS:-2}" >>"${build_log}" 2>&1 || \
   ! cpack --config "${run_root}/build/CPackConfig.cmake" >>"${build_log}" 2>&1; then
    printf 'stage\tstatus\npackage\tFAIL\n' >"${run_root}/results/summary.tsv"
else
    package_path=$(find "${run_root}/build" -type f -name '*.run' -print -quit)
    if [[ -z "${package_path}" ]]; then
        printf 'stage\tstatus\npackage\tMISSING_RUN\n' >"${run_root}/results/summary.tsv"
    else
        install_log="${run_root}/logs/install.log"
        if ! env -u ASCEND_CUSTOM_OPP_PATH bash "${package_path}" \
                --install-path="${run_root}/install" --quiet >"${install_log}" 2>&1; then
            printf 'stage\tstatus\npackage\tPASS\ninstall\tFAIL\n' >"${run_root}/results/summary.tsv"
        else
            set_env=$(find "${run_root}/install/vendors" -path '*/bin/set_env.bash' -type f -print -quit)
            custom_lib=$(find "${run_root}/install/vendors" -path '*/op_api/lib/libcust_opapi.so' -type f -print -quit)
            if [[ -z "${set_env}" || -z "${custom_lib}" ]]; then
                printf 'stage\tstatus\npackage\tPASS\ninstall\tMISSING_ARTIFACT\n' >"${run_root}/results/summary.tsv"
            else
                workspace_log="${run_root}/logs/installed-opp-workspace.log"
                # Do not preload or explicitly load a tiling .so here.  The
                # installed OPP and ASCEND_CUSTOM_OPP_PATH must resolve it.
                if env -u ASCEND_CUSTOM_OPP_PATH -u SPARSE_FLASH_ATTENTION_TILING_LIB \
                       -u LD_PRELOAD bash -c '
                           source "$1"
                           SPARSE_FLASH_ATTENTION_CUSTOM_LIB="$2" \
                           timeout "$3" python3 -u "$4" D_rope_required \
                               --workspace-only --device "$5"
                       ' bash "${set_env}" "${custom_lib}" \
                       "${SFA_910B_CASE_TIMEOUT:-180}" "${test_file}" "${device_id}" \
                       >"${workspace_log}" 2>&1; then
                    printf 'stage\tstatus\npackage\tPASS\ninstall\tPASS\nget_workspace\tPASS\n' \
                        >"${run_root}/results/summary.tsv"
                else
                    printf 'stage\tstatus\npackage\tPASS\ninstall\tPASS\nget_workspace\tFAIL\n' \
                        >"${run_root}/results/summary.tsv"
                fi
            fi
        fi
    fi
fi

archive="${TMPDIR:-/tmp}/sfa-910b-opp-install-control-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "${archive}" -C "${run_root}" results logs
echo "===== SFA 910B OPP INSTALL CONTROL SUMMARY ====="
column -t -s $'\t' "${run_root}/results/summary.tsv" 2>/dev/null || cat "${run_root}/results/summary.tsv"
echo "RESULT_ARCHIVE=${archive}"
if [[ "${SFA_910B_PUSH_RESULTS:-0}" == "1" ]]; then
    "${script_dir}/upload-910b-probe-results.sh" "${archive}"
fi

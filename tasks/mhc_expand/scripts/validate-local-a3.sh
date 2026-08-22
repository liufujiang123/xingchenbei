#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
official_source="${task_dir}/workspace/code"

fail() {
    echo "LOCAL_A3_VALIDATION_FAIL: $*" >&2
    exit 1
}

clean_managed_root() {
    local root=$1
    local tmp_base=$2
    local expected_name=mhc_expand_a3_validation
    local tmp_real root_real

    mkdir -p "${tmp_base}"
    tmp_real=$(readlink -f "${tmp_base}")
    [[ "${tmp_real}" != "/" ]] || fail "refusing to manage a directory directly below /"
    root_real=$(readlink -m "${root}")
    [[ "${root_real}" == "${tmp_real}/${expected_name}" ]] || \
        fail "unexpected managed root: ${root_real}"
    [[ ! -L "${root}" ]] || fail "managed root must not be a symlink: ${root}"

    if [[ -d "${root}" ]]; then
        find "${root}" -type d -exec chmod u+rwx {} +
        find "${root}" -type f -exec chmod u+rw {} +
        find "${root}" -depth -delete
    elif [[ -e "${root}" ]]; then
        fail "managed root exists but is not a directory: ${root}"
    fi
    mkdir -p "${root}/source" "${root}/build" "${root}/install" "${root}/logs"
}

check_adaptation_scope() {
    local mirror=$1
    local log_dir=$2
    local recursive_diff="${log_dir}/a3-adaptation.diff"
    local changed_files="${log_dir}/a3-adaptation-files.txt"
    local changed_lines="${log_dir}/a3-adaptation-changed-lines.txt"
    local diff_rc

    set +e
    diff -ru "${official_source}" "${mirror}" >"${recursive_diff}"
    diff_rc=$?
    set -e
    [[ ${diff_rc} -eq 1 ]] || {
        [[ ${diff_rc} -eq 0 ]] && fail "A3 adaptation produced no target changes"
        fail "recursive source diff failed with rc=${diff_rc}"
    }
    diff -qr "${official_source}" "${mirror}" >"${changed_files}" || true

    [[ $(wc -l <"${changed_files}") -eq 2 ]] || {
        cat "${changed_files}" >&2
        fail "A3_ADAPTATION_SCOPE_VIOLATION"
    }
    grep -Fqx "Files ${official_source}/CMakeLists.txt and ${mirror}/CMakeLists.txt differ" \
        "${changed_files}" || fail "A3_ADAPTATION_SCOPE_VIOLATION"
    grep -Fqx "Files ${official_source}/op_host/mhc_expand.cpp and ${mirror}/op_host/mhc_expand.cpp differ" \
        "${changed_files}" || fail "A3_ADAPTATION_SCOPE_VIOLATION"

    grep -E '^[+-]' "${recursive_diff}" | grep -Ev '^(---|\+\+\+)' | \
        tr -d '\r' >"${changed_lines}"
    [[ $(wc -l <"${changed_lines}") -eq 4 ]] || {
        cat "${changed_lines}" >&2
        fail "A3_ADAPTATION_SCOPE_VIOLATION"
    }
    grep -Fqx -- '-set(ASCEND_COMPUTE_UNIT ascend910b)' "${changed_lines}" || \
        fail "A3_ADAPTATION_SCOPE_VIOLATION"
    grep -Fqx -- '+set(ASCEND_COMPUTE_UNIT ascend910_93)' "${changed_lines}" || \
        fail "A3_ADAPTATION_SCOPE_VIOLATION"
    grep -Fqx -- '-                .AddConfig("ascend910b");' "${changed_lines}" || \
        fail "A3_ADAPTATION_SCOPE_VIOLATION"
    grep -Fqx -- '+                .AddConfig("ascend910_93");' "${changed_lines}" || \
        fail "A3_ADAPTATION_SCOPE_VIOLATION"

    cat "${recursive_diff}"
    echo "A3_ADAPTATION_SCOPE_PASS"
}

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -eq 0 ]] || fail "this entrypoint does not accept positional arguments"
    tmp_base=${TMPDIR:-/tmp}
    managed_root="${tmp_base%/}/mhc_expand_a3_validation"
    device_id=${MHC_EXPAND_DEVICE_ID:-4}
    [[ "${device_id}" =~ ^[0-9]+$ ]] || fail "invalid MHC_EXPAND_DEVICE_ID: ${device_id}"
    [[ "${device_id}" != "0" ]] || fail "device 0 is prohibited for local validation"
    export MHC_EXPAND_DEVICE_ID="${device_id}"
    exec "${script_dir}/with-cann85-a3.sh" \
        "${script_path}" --inside-a3-cann85 "${tmp_base}" "${managed_root}"
fi

shift
[[ $# -eq 2 ]] || fail "invalid internal invocation"
tmp_base=$1
managed_root=$2
mirror_source="${managed_root}/source"
build_dir="${managed_root}/build"
install_dir="${managed_root}/install"
log_dir="${managed_root}/logs"
package_path="${build_dir}/custom_opp_openEuler_aarch64.run"
device_id=${MHC_EXPAND_DEVICE_ID:-4}

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
[[ "${device_id}" != "0" ]] || fail "device 0 is prohibited for local validation"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-A3-ops" ]] || \
    fail "expected Ascend-cann-A3-ops, found ${ops_package:-missing}"

[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' \
    "${official_source}/CMakeLists.txt") -eq 1 ]] || \
    fail "official source compute target is not exactly ascend910b"
[[ $(grep -Fc '.AddConfig("ascend910b");' \
    "${official_source}/op_host/mhc_expand.cpp") -eq 1 ]] || \
    fail "official source OpDef target is not exactly ascend910b"
if grep -Rqs 'ascend910_93' "${official_source}"; then
    fail "official source contains local A3 target text"
fi

clean_managed_root "${managed_root}" "${tmp_base}"
cp -a "${official_source}/." "${mirror_source}/"

[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' \
    "${mirror_source}/CMakeLists.txt") -eq 1 ]] || fail "unexpected CMake target count"
[[ $(grep -Fc '.AddConfig("ascend910b");' \
    "${mirror_source}/op_host/mhc_expand.cpp") -eq 1 ]] || fail "unexpected AddConfig target count"
sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' \
    "${mirror_source}/CMakeLists.txt"
sed -i 's/\.AddConfig("ascend910b");/.AddConfig("ascend910_93");/' \
    "${mirror_source}/op_host/mhc_expand.cpp"

echo "LOCAL A3 VALIDATION ONLY"
echo "NOT FOR SUBMISSION"
echo "CANN_VERSION=${CANN_ENV_VERSION}"
echo "OPS_PACKAGE=${ops_package}"
echo "RUNTIME_SOC=Ascend910_9382"
echo "BUILD_TARGET=ascend910_93"
echo "DEVICE_ID=${device_id}"
echo "OFFICIAL_SOURCE=${official_source}"
echo "TEMPORARY_SOURCE=${mirror_source}"
echo "BUILD_DIR=${build_dir}"
echo "INSTALL_DIR=${install_dir}"

check_adaptation_scope "${mirror_source}" "${log_dir}"

cmake -S "${mirror_source}" -B "${build_dir}" \
    -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
    2>&1 | tee "${log_dir}/configure.log"
cmake --build "${build_dir}" --target binary \
    --parallel "${MHC_EXPAND_BUILD_JOBS:-2}" 2>&1 | tee "${log_dir}/binary-build.log"
cmake --build "${build_dir}" --target package \
    --parallel "${MHC_EXPAND_BUILD_JOBS:-2}" 2>&1 | tee "${log_dir}/package-build.log"

[[ -x "${package_path}" ]] || fail "A3 package was not generated: ${package_path}"
[[ -f "${build_dir}/op_host/libcust_opapi.so" ]] || \
    fail "generated ACLNN library is missing"
mapfile -t build_indexes < <(find "${build_dir}/op_kernel/ascendc_kernels/binary/config/ascend910_93" \
    -type f -name binary_info_config.json -print 2>/dev/null)
[[ ${#build_indexes[@]} -eq 1 ]] || \
    fail "expected one ascend910_93 build index, found ${#build_indexes[@]}"
grep -q 'MhcExpand' "${build_indexes[0]}" || fail "A3 build index does not contain MhcExpand"
if find "${build_dir}/op_kernel/ascendc_kernels/binary/config/ascend910b" \
    -type f -name binary_info_config.json -print -quit 2>/dev/null | grep -q .; then
    fail "A3 build unexpectedly produced an ascend910b MhcExpand index"
fi

extract_home="${managed_root}/extract-home"
mkdir -p "${extract_home}"
env HOME="${extract_home}" "${package_path}" --quiet --install-path="${install_dir}" \
    2>&1 | tee "${log_dir}/install.log"
mapfile -t installed_indexes < <(find "${install_dir}" -type f \
    -path '*/kernel/config/ascend910_93/binary_info_config.json' -print)
[[ ${#installed_indexes[@]} -eq 1 ]] || \
    fail "expected one installed ascend910_93 index, found ${#installed_indexes[@]}"
grep -q 'MhcExpand' "${installed_indexes[0]}" || \
    fail "installed A3 index does not contain MhcExpand"

custom_set_env="${install_dir}/vendors/custom/bin/set_env.bash"
[[ -r "${custom_set_env}" ]] || fail "custom OPP environment script is missing"
set +u
# shellcheck disable=SC1090
source "${custom_set_env}"
set -u
export MHC_EXPAND_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"
[[ -r "${MHC_EXPAND_CUSTOM_LIB}" ]] || fail "custom ACLNN library is missing"
python -u -c 'import ctypes, os; p=os.path.realpath(os.environ["MHC_EXPAND_CUSTOM_LIB"]); ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL); print(f"CUSTOM_ACLNN_LIBRARY_LOAD_PASS={p}", flush=True)' \
    2>&1 | tee "${log_dir}/library-load.log"

phases=(
    forward_fp16_smoke
    forward_bf16_smoke
    forward_boundary
    backward_fp16_smoke
    backward_bf16_smoke
    backward_boundary
)
for phase in "${phases[@]}"; do
    echo "CORRECTNESS_PHASE=${phase}"
    timeout "${MHC_EXPAND_PHASE_TIMEOUT:-300}" \
        python -u "${task_dir}/tests/run_staged_correctness.py" --phase "${phase}" \
        2>&1 | tee "${log_dir}/${phase}.log"
done

timeout "${MHC_EXPAND_VALIDATE_TIMEOUT:-900}" \
    python -m pytest "${task_dir}/tests/test_mhc_expand.py" -v \
    2>&1 | tee "${log_dir}/full-correctness.log"

echo "PACKAGE=${package_path}"
echo "INSTALLED_INDEX=${installed_indexes[0]}"
echo "LOCAL_A3_BUILD_PASS"
echo "LOCAL_A3_CORRECTNESS_PASS"

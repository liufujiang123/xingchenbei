#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"

fail() {
    echo "PLATFORM_910B_BUILD_FAIL: $*" >&2
    exit 1
}

clean_managed_root() {
    local root=$1
    local tmp_base=$2
    local expected_name=mhc_expand_platform_910b
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
    mkdir -p "${root}/build" "${root}/logs"
}

if [[ "${1:-}" != "--inside-platform-cann85" ]]; then
    [[ $# -eq 0 ]] || fail "this entrypoint does not accept positional arguments"
    tmp_base=${TMPDIR:-/tmp}
    managed_root="${tmp_base%/}/mhc_expand_platform_910b"
    export CANN85_EXPECTED_OPS_PACKAGE=Ascend-cann-910b-ops
    exec "${script_dir}/with-cann85.sh" \
        "${script_path}" --inside-platform-cann85 "${tmp_base}" "${managed_root}"
fi

shift
[[ $# -eq 2 ]] || fail "invalid internal invocation"
tmp_base=$1
managed_root=$2
build_dir="${managed_root}/build"
log_dir="${managed_root}/logs"
package_path="${build_dir}/custom_opp_openEuler_aarch64.run"

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-910b-ops" ]] || \
    fail "expected Ascend-cann-910b-ops, found ${ops_package:-missing}"

[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' \
    "${source_dir}/CMakeLists.txt") -eq 1 ]] || \
    fail "official source does not have exactly one ascend910b compute target"
[[ $(grep -Fc '.AddConfig("ascend910b");' \
    "${source_dir}/op_host/mhc_expand.cpp") -eq 1 ]] || \
    fail "official source does not have exactly one ascend910b OpDef target"
if grep -Rqs 'ascend910_93' "${source_dir}"; then
    fail "official source contains local A3 target text"
fi

clean_managed_root "${managed_root}" "${tmp_base}"

echo "PLATFORM 910B BUILD"
echo "CANN_VERSION=${CANN_ENV_VERSION}"
echo "OPS_PACKAGE=${ops_package}"
echo "TARGET=ascend910b"
echo "SOURCE=${source_dir}"
echo "BUILD_DIR=${build_dir}"

MHC_EXPAND_BUILD_DIR="${build_dir}" \
MHC_EXPAND_BUILD_JOBS="${MHC_EXPAND_BUILD_JOBS:-2}" \
    "${script_dir}/build.sh" 2>&1 | tee "${log_dir}/build.log"

[[ -x "${package_path}" ]] || fail "package was not generated: ${package_path}"
[[ -f "${build_dir}/op_host/libcust_opapi.so" ]] || \
    fail "generated ACLNN library is missing"
[[ -f "${build_dir}/op_host/libcust_opmaster_rt2.0.so" ]] || \
    fail "generated Host tiling library is missing"
mapfile -t kernel_objects < <(find "${build_dir}/op_kernel/ascendc_kernels/binary/ascend910b" \
    -type f -name 'MhcExpand_*.o' -print 2>/dev/null)
[[ ${#kernel_objects[@]} -eq 2 ]] || \
    fail "expected two FP16/BF16 kernel objects, found ${#kernel_objects[@]}"
[[ -r "${build_dir}/op_kernel/ascendc_kernels/binary/config/ascend910b/binary_info_config.json" ]] || \
    fail "ascend910b binary-info index is missing"

echo "HOST_BUILD=PASS"
echo "FP16_KERNEL_BUILD=PASS"
echo "BF16_KERNEL_BUILD=PASS"
echo "ACLNN_LIBRARY_BUILD=PASS"
echo "PACKAGE=${package_path}"
echo "PLATFORM_910B_BUILD_PASS"

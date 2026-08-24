#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
fail() { echo "SPARSE_FLASH_ATTENTION_PLATFORM_BUILD_FAIL: $*" >&2; exit 1; }

if [[ "${1:-}" != "--inside-platform-cann85" ]]; then
    [[ $# -eq 0 ]] || fail "this entrypoint does not accept arguments"
    export CANN85_EXPECTED_OPS_PACKAGE=Ascend-cann-910b-ops
    exec "${script_dir}/with-cann85.sh" "${script_path}" --inside-platform-cann85
fi
shift
[[ $# -eq 0 ]] || fail "invalid internal invocation"

"${script_dir}/check-cann85.sh"
[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' "${source_dir}/CMakeLists.txt") -eq 1 ]] || fail "CMake target drift"
[[ $(grep -Fc '.AddConfig("ascend910b");' "${source_dir}/op_host/sparse_flash_attention.cpp") -eq 1 ]] || fail "OpDef target drift"
! grep -Rqs 'ascend910_93' "${source_dir}" || fail "submission source contains local A3 target"

if [[ -n "${SPARSE_FLASH_ATTENTION_PLATFORM_BUILD_DIR:-}" ]]; then
    build_dir=${SPARSE_FLASH_ATTENTION_PLATFORM_BUILD_DIR}
else
    build_root=$(mktemp -d "${TMPDIR:-/tmp}/sparse_flash_attention_platform_910b.XXXXXX")
    build_dir="${build_root}/build"
fi
mkdir -p "${build_dir}"
cmake -S "${source_dir}" -B "${build_dir}" -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")"
build_log="${build_dir}/build.log"
set +e
cmake --build "${build_dir}" --target binary --parallel "${SPARSE_FLASH_ATTENTION_BUILD_JOBS:-2}" 2>&1 | tee "${build_log}"
build_status=${PIPESTATUS[0]}
set -e
[[ ${build_status} -eq 0 ]] || exit "${build_status}"
! grep -Eq 'Opc tool compile failed|Opc tool compile process failed|errors generated\.' "${build_log}" || fail "kernel compiler reported an error"
cmake --build "${build_dir}" --parallel "${SPARSE_FLASH_ATTENTION_BUILD_JOBS:-2}"

[[ -f "${build_dir}/libcust_opapi.so" ]] || fail "ACLNN library is missing"
[[ -f "${build_dir}/op_host/libcustom_ascendc_cust_optiling.so" ]] || fail "Host tiling library is missing"
mapfile -t kernel_objects < <(find "${build_dir}/op_kernel/ascendc_kernels/binary/ascend910b" -type f -name 'SparseFlashAttention_*.o' -print 2>/dev/null)
[[ ${#kernel_objects[@]} -eq 3 ]] || fail "expected FP16, FP32, and BF16 binaries, found ${#kernel_objects[@]}"
echo "HOST_BUILD=PASS"
echo "KERNEL_BINARY_COUNT=${#kernel_objects[@]}"
echo "SPARSE_FLASH_ATTENTION_PLATFORM_910B_BUILD_PASS"

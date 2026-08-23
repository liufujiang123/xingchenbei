#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
repo_root=$(cd "${task_dir}/../.." && pwd)
official_source="${task_dir}/workspace/code"
wrapper="${script_dir}/with-cann85-a3.sh"

fail() {
    echo "SPARSE_FLASH_ATTENTION_LOCAL_A3_FAIL: $*" >&2
    exit 1
}

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -le 1 ]] || fail "usage: $0 [case-name]"
    case_name=${1:-all}
    export SPARSE_FLASH_ATTENTION_DEVICE_ID="${SPARSE_FLASH_ATTENTION_DEVICE_ID:-4}"
    exec "${wrapper}" "${script_path}" --inside-a3-cann85 "${case_name}"
fi

shift
[[ $# -eq 1 ]] || fail "invalid internal invocation"
case_name=$1
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-4}
[[ "${device_id}" =~ ^[0-9]+$ ]] || fail "invalid device id: ${device_id}"
[[ "${device_id}" != "0" ]] || fail "device 0 is prohibited by the proven local flow"
[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"

managed_root="${TMPDIR:-/tmp}/sparse_flash_attention_a3_validation"
mirror_source="${managed_root}/source"
build_dir="${managed_root}/build"
log_dir="${managed_root}/logs"

if [[ -d "${managed_root}" ]]; then
    find "${managed_root}" -type d -exec chmod u+rwx {} +
    find "${managed_root}" -type f -exec chmod u+rw {} +
    find "${managed_root}" -depth -delete
elif [[ -e "${managed_root}" ]]; then
    fail "managed root exists but is not a directory: ${managed_root}"
fi
mkdir -p "${mirror_source}" "${build_dir}" "${log_dir}"
cp -a "${official_source}/." "${mirror_source}/"

[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' "${mirror_source}/CMakeLists.txt") -eq 1 ]] || \
    fail "unexpected official compute target"
[[ $(grep -Fc '.AddConfig("ascend910b");' "${mirror_source}/op_host/sparse_flash_attention.cpp") -eq 1 ]] || \
    fail "unexpected official OpDef target"
sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' \
    "${mirror_source}/CMakeLists.txt"
sed -i 's/\.AddConfig("ascend910b");/.AddConfig("ascend910_93");/' \
    "${mirror_source}/op_host/sparse_flash_attention.cpp"

changed_files="${log_dir}/adaptation-files.txt"
diff -qr "${official_source}" "${mirror_source}" >"${changed_files}" || true
[[ $(wc -l <"${changed_files}") -eq 2 ]] || {
    cat "${changed_files}" >&2
    fail "temporary A3 adaptation changed unexpected files"
}

echo "LOCAL_A3_VALIDATION_ONLY"
echo "CANN_VERSION=${CANN_ENV_VERSION}"
echo "TARGET=ascend910_93"
echo "DEVICE_ID=${device_id}"
echo "OFFICIAL_SOURCE=${official_source}"
echo "TEMPORARY_SOURCE=${mirror_source}"
echo "BUILD_DIR=${build_dir}"

cmake -S "${mirror_source}" -B "${build_dir}" \
    -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
    2>&1 | tee "${log_dir}/configure.log"
cmake --build "${build_dir}" --parallel 2 2>&1 | tee "${log_dir}/build.log"

custom_lib="${build_dir}/libcust_opapi.so"
[[ -r "${custom_lib}" ]] || fail "generated ACLNN library is missing: ${custom_lib}"
export SPARSE_FLASH_ATTENTION_CUSTOM_LIB="${custom_lib}"
python -c 'import ctypes, os; ctypes.CDLL(os.environ["SPARSE_FLASH_ATTENTION_CUSTOM_LIB"], mode=ctypes.RTLD_GLOBAL); print("CUSTOM_SHARED_LIBRARY_LOAD_PASS=" + os.path.realpath(os.environ["SPARSE_FLASH_ATTENTION_CUSTOM_LIB"]))'

timeout "${SPARSE_FLASH_ATTENTION_VALIDATE_TIMEOUT:-300}" \
    python -u "${task_dir}/tests/test_sparse_flash_attention.py" "${case_name}" --device "${device_id}" \
    2>&1 | tee "${log_dir}/correctness.log"

echo "CUSTOM_SHARED_LIBRARY=${custom_lib}"
echo "CORRECTNESS_LOG=${log_dir}/correctness.log"
echo "SPARSE_FLASH_ATTENTION_LOCAL_A3_PASS"

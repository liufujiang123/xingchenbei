#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
official_source="${task_dir}/workspace/code"
fail() { echo "LOCAL_A3_VALIDATION_FAIL: $*" >&2; exit 1; }

if [[ "${1:-}" != "--inside-a3-cann85" ]]; then
    [[ $# -eq 0 ]] || fail "this entrypoint does not accept positional arguments"
    managed_root="${TMPDIR:-/tmp}/mhc_sinkhorn_a3_validation"
    export MHC_SINKHORN_DEVICE_ID="${MHC_SINKHORN_DEVICE_ID:-4}"
    exec "${script_dir}/with-cann85-a3.sh" "${script_path}" --inside-a3-cann85 "${managed_root}"
fi
shift
[[ $# -eq 1 ]] || fail "invalid internal invocation"
managed_root=$1
expected_root="${TMPDIR:-/tmp}/mhc_sinkhorn_a3_validation"
[[ "$(readlink -m "${managed_root}")" == "$(readlink -m "${expected_root}")" ]] || fail "unexpected managed root"
[[ ! -L "${managed_root}" ]] || fail "managed root must not be a symlink"
device_id=${MHC_SINKHORN_DEVICE_ID:-4}
[[ "${device_id}" =~ ^[0-9]+$ && "${device_id}" != "0" ]] || fail "use a valid nonzero device"
[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-A3-ops" ]] || fail "expected Ascend-cann-A3-ops"

runtime_soc=$(python - "${device_id}" <<'PY'
import ctypes
import sys

device = int(sys.argv[1])
lib = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
lib.aclInit.argtypes = [ctypes.c_char_p]
lib.aclInit.restype = ctypes.c_int
lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
lib.aclrtSetDevice.restype = ctypes.c_int
lib.aclrtGetSocName.restype = ctypes.c_char_p
lib.aclrtResetDevice.argtypes = [ctypes.c_int32]
lib.aclFinalize.restype = ctypes.c_int
if lib.aclInit(None) != 0 or lib.aclrtSetDevice(device) != 0:
    raise SystemExit(1)
print(lib.aclrtGetSocName().decode())
lib.aclrtResetDevice(device)
lib.aclFinalize()
PY
)
[[ "${runtime_soc}" == "Ascend910_9382" ]] || fail "expected local A3 SOC, found ${runtime_soc}"

if [[ -d "${managed_root}" ]]; then
    find "${managed_root}" -type d -exec chmod u+rwx {} +
    find "${managed_root}" -type f -exec chmod u+rw {} +
    find "${managed_root}" -depth -delete
elif [[ -e "${managed_root}" ]]; then
    fail "managed root exists but is not a directory"
fi
mkdir -p "${managed_root}/source" "${managed_root}/build" "${managed_root}/install" "${managed_root}/logs"
mirror_source="${managed_root}/source"
build_dir="${managed_root}/build"
install_dir="${managed_root}/install"
log_dir="${managed_root}/logs"
package_path="${build_dir}/custom_opp_openEuler_aarch64.run"
cp -a "${official_source}/." "${mirror_source}/"
sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' "${mirror_source}/CMakeLists.txt"
sed -i 's/\.AddConfig("ascend910b");/.AddConfig("ascend910_93");/' "${mirror_source}/op_host/mhc_sinkhorn.cpp"

diff_output="${log_dir}/a3-adaptation.diff"
set +e
diff -ru "${official_source}" "${mirror_source}" >"${diff_output}"
diff_rc=$?
set -e
[[ ${diff_rc} -eq 1 ]] || fail "A3 adaptation must change exactly the target declarations"
changed_files=$(diff -qr "${official_source}" "${mirror_source}" || true)
[[ $(wc -l <<<"${changed_files}") -eq 2 ]] || fail "A3 adaptation escaped target declarations"
grep -q 'CMakeLists.txt' <<<"${changed_files}" || fail "CMake target adaptation missing"
grep -q 'op_host/mhc_sinkhorn.cpp' <<<"${changed_files}" || fail "OpDef target adaptation missing"

echo "LOCAL A3 VALIDATION ONLY"
echo "NOT FOR SUBMISSION"
echo "CANN_VERSION=${CANN_ENV_VERSION}"
echo "OPS_PACKAGE=${ops_package}"
echo "RUNTIME_SOC=${runtime_soc}"
echo "BUILD_TARGET=ascend910_93"
echo "DEVICE_ID=${device_id}"

cmake -S "${mirror_source}" -B "${build_dir}" -DASCEND_CANN_PACKAGE_PATH="$(readlink -f "${ASCEND_HOME_PATH}")" \
    2>&1 | tee "${log_dir}/configure.log"
cmake --build "${build_dir}" --target binary --parallel "${MHC_SINKHORN_BUILD_JOBS:-2}" \
    2>&1 | tee "${log_dir}/binary-build.log"
cmake --build "${build_dir}" --target package --parallel "${MHC_SINKHORN_BUILD_JOBS:-2}" \
    2>&1 | tee "${log_dir}/package-build.log"
[[ -x "${package_path}" ]] || fail "A3 package was not generated"
[[ -f "${build_dir}/op_host/libcust_opapi.so" ]] || fail "generated ACLNN library is missing"

extract_home="${managed_root}/extract-home"
mkdir -p "${extract_home}"
env HOME="${extract_home}" "${package_path}" --quiet --install-path="${install_dir}" \
    2>&1 | tee "${log_dir}/install.log"
custom_set_env="${install_dir}/vendors/custom/bin/set_env.bash"
[[ -r "${custom_set_env}" ]] || fail "custom OPP environment script is missing"
set +u
# shellcheck disable=SC1090
source "${custom_set_env}"
set -u
export MHC_SINKHORN_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"
[[ -r "${MHC_SINKHORN_CUSTOM_LIB}" ]] || fail "custom ACLNN library is missing"
source_digest=$(find "${official_source}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d " " -f 1)
echo "SOURCE_SHA256=${source_digest}" > "${log_dir}/source.sha256"

python -m pytest "${task_dir}/tests/test_reference_mhc_sinkhorn.py" \
    "${task_dir}/tests/test_source_contract.py" -q \
    2>&1 | tee "${log_dir}/cpu-reference.log"
timeout "${MHC_SINKHORN_VALIDATE_TIMEOUT:-900}" \
    python -m pytest "${task_dir}/tests/test_mhc_sinkhorn_acl.py" -v \
    2>&1 | tee "${log_dir}/npu-correctness.log"
echo "LOCAL_A3_BUILD_PASS"
echo "LOCAL_A3_CORRECTNESS_PASS"

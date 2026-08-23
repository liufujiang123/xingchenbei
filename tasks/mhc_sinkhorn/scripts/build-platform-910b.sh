#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
fail() { echo "PLATFORM_910B_BUILD_FAIL: $*" >&2; exit 1; }

if [[ "${1:-}" != "--inside-platform-cann85" ]]; then
    [[ $# -eq 0 ]] || fail "this entrypoint does not accept positional arguments"
    tmp_base=${TMPDIR:-/tmp}
    managed_root="${tmp_base%/}/mhc_sinkhorn_platform_910b"
    export CANN85_EXPECTED_OPS_PACKAGE=Ascend-cann-910b-ops
    exec "${script_dir}/with-cann85.sh" "${script_path}" --inside-platform-cann85 "${managed_root}"
fi
shift
[[ $# -eq 1 ]] || fail "invalid internal invocation"
managed_root=$1
expected_root="${TMPDIR:-/tmp}/mhc_sinkhorn_platform_910b"
[[ "$(readlink -m "${managed_root}")" == "$(readlink -m "${expected_root}")" ]] || fail "unexpected managed root"
[[ ! -L "${managed_root}" ]] || fail "managed root must not be a symlink"
mkdir -p "${managed_root}/build" "${managed_root}/logs"
build_dir="${managed_root}/build"
log_dir="${managed_root}/logs"
package_path="${build_dir}/custom_opp_openEuler_aarch64.run"

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN 8.5.0 is not active"
environment_report=$("${script_dir}/check-cann85.sh")
ops_package=$(sed -n 's/^OPS_PACKAGE=//p' <<<"${environment_report}" | head -1)
[[ "${ops_package}" == "Ascend-cann-910b-ops" ]] || fail "expected 910B ops package"
[[ $(grep -Fc 'set(ASCEND_COMPUTE_UNIT ascend910b)' "${source_dir}/CMakeLists.txt") -eq 1 ]] || fail "CMake target drift"
[[ $(grep -Fc '.AddConfig("ascend910b");' "${source_dir}/op_host/mhc_sinkhorn.cpp") -eq 1 ]] || fail "OpDef target drift"
! grep -Rqs 'ascend910_93' "${source_dir}" || fail "official source contains local A3 target"

echo "PLATFORM 910B BUILD"
echo "CANN_VERSION=${CANN_ENV_VERSION}"
echo "OPS_PACKAGE=${ops_package}"
echo "TARGET=ascend910b"
MHC_SINKHORN_BUILD_DIR="${build_dir}" MHC_SINKHORN_BUILD_JOBS="${MHC_SINKHORN_BUILD_JOBS:-2}" \
    "${script_dir}/build.sh" 2>&1 | tee "${log_dir}/build.log"

[[ -x "${package_path}" ]] || fail "package was not generated"
[[ -f "${build_dir}/op_host/libcust_opapi.so" ]] || fail "ACLNN library is missing"
[[ -f "${build_dir}/op_host/libcust_opmaster_rt2.0.so" ]] || fail "Host tiling library is missing"
mapfile -t kernel_objects < <(find "${build_dir}/op_kernel/ascendc_kernels/binary/ascend910b" -type f -name 'MhcSinkhorn_*.o' -print 2>/dev/null)
[[ ${#kernel_objects[@]} -eq 2 ]] || fail "expected two dtype binaries, found ${#kernel_objects[@]}"
python - "${build_dir}" <<'PY' || fail "compiled binary does not contain all N/mask template regimes"
import glob
import json
import os
import sys

paths = glob.glob(os.path.join(sys.argv[1], "op_kernel", "ascendc_kernels", "binary", "ascend910b", "mhc_sinkhorn", "MhcSinkhorn_*.json"))
if len(paths) != 2:
    raise SystemExit(1)
for path in paths:
    with open(path, encoding="utf-8") as source:
        if len(json.load(source).get("kernelList", [])) < 9:
            raise SystemExit(1)
PY
[[ -r "${build_dir}/op_kernel/ascendc_kernels/binary/config/ascend910b/binary_info_config.json" ]] || fail "binary-info index is missing"
echo "HOST_BUILD=PASS"
echo "FP16_KERNEL_BUILD=PASS"
echo "FP32_KERNEL_BUILD=PASS"
echo "PACKAGE=${package_path}"
echo "PLATFORM_910B_BUILD_PASS"

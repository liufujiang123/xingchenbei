#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd "${script_dir}/.." && pwd)
source_dir="${task_dir}/workspace/code"
build_dir="${MHC_SINKHORN_BUILD_DIR:-${TMPDIR:-/tmp}/mhc_sinkhorn_build_cann850}"

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" && -n "${CANN85_HOME:-}" ]] || {
    echo "CANN 8.5.0 is not selected. Run through scripts/with-cann85.sh." >&2
    exit 2
}
"${script_dir}/check-cann85.sh" >/dev/null
cann_home_real=$(readlink -f "${CANN85_HOME}")
ascend_home_real=$(readlink -f "${ASCEND_HOME_PATH}")
case "${ascend_home_real}/" in
    "${cann_home_real}/"*) ;;
    *) echo "ASCEND_HOME_PATH is outside CANN85_HOME" >&2; exit 2 ;;
esac

if [[ -f "${build_dir}/CMakeCache.txt" ]]; then
    cached_cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "${build_dir}/CMakeCache.txt" | head -1)
    if [[ -n "${cached_cann}" && "$(readlink -f "${cached_cann}")" != "${ascend_home_real}" ]]; then
        echo "Build directory contains another CANN cache; choose a new MHC_SINKHORN_BUILD_DIR." >&2
        exit 2
    fi
fi

original_opp_path="${ASCEND_OPP_PATH:-}"
opp_view=""
cleanup() {
    if [[ -n "${opp_view}" && -d "${opp_view}" ]]; then
        find "${opp_view}" -type l -delete
        find "${opp_view}" -depth -type d -empty -delete
    fi
}
trap cleanup EXIT
if [[ -n "${original_opp_path}" && -e "${original_opp_path}/vendors/config.ini" && ! -r "${original_opp_path}/vendors/config.ini" ]]; then
    opp_view=$(mktemp -d "${TMPDIR:-/tmp}/mhc_sinkhorn_oppview.XXXXXX")
    mkdir -p "${opp_view}/op_impl"
    ln -s "${original_opp_path}/built-in/op_impl" "${opp_view}/op_impl/built-in"
    ln -s "${original_opp_path}/scene.info" "${opp_view}/scene.info"
    export ASCEND_OPP_PATH="${opp_view}"
fi

cmake -S "${source_dir}" -B "${build_dir}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home_real}"
binary_log="${build_dir}/mhc_sinkhorn_binary_build.log"
set +e
cmake --build "${build_dir}" --target binary --parallel "${MHC_SINKHORN_BUILD_JOBS:-2}" 2>&1 | tee "${binary_log}"
binary_status=${PIPESTATUS[0]}
set -e
[[ ${binary_status} -eq 0 ]] || exit "${binary_status}"
if grep -Eq 'Opc tool compile failed|Opc tool compile process failed|errors generated\.' "${binary_log}"; then
    echo "Kernel compiler reported an error even though the outer build returned success." >&2
    exit 1
fi
cmake --build "${build_dir}" --target package --parallel "${MHC_SINKHORN_BUILD_JOBS:-2}"
echo "BUILD_DIR=${build_dir}"
echo "PACKAGE=${build_dir}/custom_opp_openEuler_aarch64.run"

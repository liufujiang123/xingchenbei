#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_dir=$(cd "${script_dir}/.." && pwd)
build_dir="${MHC_EXPAND_BUILD_DIR:-${TMPDIR:-/tmp}/mhc_expand_build_cann850}"
install_dir="${MHC_EXPAND_INSTALL_DIR:-${TMPDIR:-/tmp}/mhc_expand_install_cann850}"
package_path="${build_dir}/custom_opp_openEuler_aarch64.run"

if [[ "${CANN_ENV_VERSION:-}" != "8.5.0" || -z "${CANN85_HOME:-}" ]]; then
    echo "CANN 8.5.0 is not selected. Run through scripts/with-cann85.sh." >&2
    exit 2
fi
"${script_dir}/check-cann85.sh" >/dev/null

if [[ ! -x "${package_path}" ]]; then
    echo "Missing package: ${package_path}; run scripts/build.sh first" >&2
    exit 2
fi

timeout "${MHC_EXPAND_RUNTIME_TIMEOUT:-60}" \
    python "${task_dir}/tests/acl_runtime_probe.py" \
    --device "${MHC_EXPAND_DEVICE_ID:-0}"

extract_home="${TMPDIR:-/tmp}/mhc_expand_package_extract_home"
mkdir -p "${extract_home}"
env HOME="${extract_home}" "${package_path}" --quiet --install-path="${install_dir}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
# shellcheck disable=SC1090
source "${install_dir}/vendors/custom/bin/set_env.bash"
export MHC_EXPAND_CUSTOM_LIB="${install_dir}/vendors/custom/op_api/lib/libcust_opapi.so"

timeout "${MHC_EXPAND_VALIDATE_TIMEOUT:-300}" \
    python -m pytest "${task_dir}/tests/test_mhc_expand.py" -v

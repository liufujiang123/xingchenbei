#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

snapshot() {
    for var_name in ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH \
        ASCEND_CUSTOM_OPP_PATH ASCEND_TOOLKIT_HOME CANN_INSTALL_PATH \
        LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH PATH; do
        printf '%s=%s\n' "${var_name}" "${!var_name-}"
    done
}

before=$(snapshot)
echo "PARENT_BEFORE_ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"
"${script_dir}/with-cann85.sh" "${script_dir}/check-cann85.sh"
after=$(snapshot)
echo "PARENT_AFTER_ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"

if [[ "${before}" != "${after}" ]]; then
    echo "Parent environment changed across with-cann85.sh" >&2
    exit 1
fi
echo "PARENT_ENVIRONMENT_UNCHANGED"

#!/usr/bin/env bash
set -euo pipefail

fail() { echo "CANN85 environment check failed: $*" >&2; exit 1; }

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN_ENV_VERSION is not 8.5.0"
[[ -n "${CANN85_HOME:-}" && -d "${CANN85_HOME}" ]] || fail "CANN85_HOME is invalid"
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || fail "ASCEND_HOME_PATH is invalid"
[[ -n "${ASCEND_OPP_PATH:-}" && -d "${ASCEND_OPP_PATH}" ]] || fail "ASCEND_OPP_PATH is invalid"
cann_home_real=$(readlink -f "${CANN85_HOME}")
for path_name in ASCEND_HOME_PATH ASCEND_OPP_PATH; do
    path_value=$(readlink -f "${!path_name}")
    case "${path_value}/" in
        "${cann_home_real}/"*) ;;
        *) fail "${path_name} is outside CANN85_HOME" ;;
    esac
done
toolkit_info=$(find "${cann_home_real}" -type f -name ascend_toolkit_install.info -print -quit)
ops_info=$(find "${cann_home_real}" -type f -name ascend_ops_install.info -print -quit)
[[ -r "${toolkit_info}" && -r "${ops_info}" ]] || fail "Toolkit or ops metadata is missing"
[[ "$(sed -n 's/^version=//p' "${toolkit_info}" | head -1)" == "8.5.0" ]] || fail "Toolkit version mismatch"
[[ "$(sed -n 's/^version=//p' "${ops_info}" | head -1)" == "8.5.0" ]] || fail "ops version mismatch"
ops_package=$(sed -n 's/^package_name=//p' "${ops_info}" | head -1)
[[ -z "${CANN85_EXPECTED_OPS_PACKAGE:-}" || "${ops_package}" == "${CANN85_EXPECTED_OPS_PACKAGE}" ]] || \
    fail "ops package mismatch: expected ${CANN85_EXPECTED_OPS_PACKAGE}, found ${ops_package:-missing}"
for compiler_name in ccec bisheng; do
    compiler=$(command -v "${compiler_name}" || true)
    [[ -n "${compiler}" ]] || fail "${compiler_name} is not on PATH"
    case "$(readlink -f "${compiler}")/" in
        "${cann_home_real}/"*) ;;
        *) fail "${compiler_name} is outside CANN85_HOME" ;;
    esac
done
echo "CANN_ENV_VERSION=${CANN_ENV_VERSION}"
echo "OPS_PACKAGE=${ops_package}"
echo "CANN85_ENVIRONMENT_OK"

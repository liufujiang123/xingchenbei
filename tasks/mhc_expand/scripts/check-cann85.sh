#!/usr/bin/env bash
set -euo pipefail

fail() {
    echo "CANN85 environment check failed: $*" >&2
    exit 1
}

[[ "${CANN_ENV_VERSION:-}" == "8.5.0" ]] || fail "CANN_ENV_VERSION is not 8.5.0"
[[ -n "${CANN85_HOME:-}" && -d "${CANN85_HOME}" ]] || fail "CANN85_HOME is invalid"
[[ -n "${CANN85_SET_ENV:-}" && -r "${CANN85_SET_ENV}" ]] || fail "CANN85_SET_ENV is invalid"
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || fail "ASCEND_HOME_PATH is invalid"
[[ -n "${ASCEND_OPP_PATH:-}" && -d "${ASCEND_OPP_PATH}" ]] || fail "ASCEND_OPP_PATH is invalid"

cann_home_real=$(readlink -f "${CANN85_HOME}")
for path_name in ASCEND_HOME_PATH ASCEND_OPP_PATH; do
    path_value=$(readlink -f "${!path_name}")
    case "${path_value}/" in
        "${cann_home_real}/"*) ;;
        *) fail "${path_name} is outside CANN85_HOME: ${path_value}" ;;
    esac
done

toolkit_info=$(find "${cann_home_real}" -type f -name ascend_toolkit_install.info -print -quit)
ops_info=$(find "${cann_home_real}" -type f -name ascend_ops_install.info -print -quit)
[[ -r "${toolkit_info}" ]] || fail "Toolkit metadata is missing"
[[ -r "${ops_info}" ]] || fail "ops metadata is missing"
[[ "$(sed -n 's/^version=//p' "${toolkit_info}" | head -1)" == "8.5.0" ]] || fail "Toolkit version mismatch"
[[ "$(sed -n 's/^version=//p' "${ops_info}" | head -1)" == "8.5.0" ]] || fail "ops version mismatch"
ops_package=$(sed -n 's/^package_name=//p' "${ops_info}" | head -1)
if [[ -n "${CANN85_EXPECTED_OPS_PACKAGE:-}" &&
      "${ops_package}" != "${CANN85_EXPECTED_OPS_PACKAGE}" ]]; then
    fail "ops package mismatch: expected ${CANN85_EXPECTED_OPS_PACKAGE}, found ${ops_package:-missing}"
fi

ccec=$(command -v ccec || true)
bisheng=$(command -v bisheng || true)
[[ -n "${ccec}" ]] || fail "ccec is not on PATH"
[[ -n "${bisheng}" ]] || fail "bisheng is not on PATH"
ccec_real=$(readlink -f "${ccec}")
bisheng_real=$(readlink -f "${bisheng}")
for compiler in "${ccec_real}" "${bisheng_real}"; do
    case "${compiler}/" in
        "${cann_home_real}/"*) ;;
        *) fail "compiler is outside CANN85_HOME: ${compiler}" ;;
    esac
done

acl_header=$(find "${cann_home_real}" -type f -path '*/include/acl/acl.h' -print -quit)
[[ -n "${acl_header}" ]] || fail "ACL header is missing"
include_dir=$(dirname "${acl_header}")
include_dir=$(dirname "${include_dir}")
ascendcl=$(find "${cann_home_real}" -type f -name 'libascendcl.so' -print -quit)
[[ -d "${include_dir}" ]] || fail "ACL include directory is missing"
[[ -n "${ascendcl}" ]] || fail "libascendcl.so is missing"

for var_name in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
    var_value=${!var_name-}
    grep -Eq "/usr/local/Ascend/(cann-9|ascend-toolkit|opp|compiler)(/|:|$)" <<<"${var_value}" && \
        fail "CANN 9 path leakage in ${var_name}"
done

loaded_ascendcl_report=$(python - <<'PY'
import ctypes

ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
with open("/proc/self/maps", encoding="utf-8") as maps:
    paths = sorted({line.split()[-1] for line in maps if "libascendcl.so" in line})
print("CANN85_RESOLVED_LIBASCENDCL=" + (paths[0] if paths else ""))
PY
)
loaded_ascendcl=$(sed -n 's/^CANN85_RESOLVED_LIBASCENDCL=//p' \
    <<<"${loaded_ascendcl_report}" | tail -1)
[[ -n "${loaded_ascendcl}" ]] || fail "libascendcl.so could not be resolved"
case "$(readlink -f "${loaded_ascendcl}")/" in
    "${cann_home_real}/"*) ;;
    *) fail "libascendcl.so resolved outside CANN85_HOME: ${loaded_ascendcl}" ;;
esac

compiler_ldd=$(ldd "${ccec_real}" 2>&1 || true)
grep -Eq "/usr/local/Ascend/(cann-9|ascend-toolkit|opp|compiler)(/|:|$)" \
    <<<"${compiler_ldd}" && fail "ccec dynamic dependency resolves to CANN 9"

echo "CANN_ENV_VERSION=${CANN_ENV_VERSION}"
echo "CANN_HOME=${cann_home_real}"
echo "CANN85_SET_ENV=$(readlink -f "${CANN85_SET_ENV}")"
echo "ASCEND_HOME_PATH=$(readlink -f "${ASCEND_HOME_PATH}")"
echo "ASCEND_OPP_PATH=$(readlink -f "${ASCEND_OPP_PATH}")"
echo "OPS_PACKAGE=${ops_package}"
echo "CCEC=${ccec_real}"
echo "BISHENG=${bisheng_real}"
echo "CMAKE=$(readlink -f "$(command -v cmake)")"
echo "PYTHON_COMMAND=$(command -v python)"
echo "PYTHON_REAL=$(readlink -f "$(command -v python)")"
echo "INCLUDE=${include_dir}"
echo "LIBASCENDCL=$(readlink -f "${loaded_ascendcl}")"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "CANN85_ENVIRONMENT_OK"

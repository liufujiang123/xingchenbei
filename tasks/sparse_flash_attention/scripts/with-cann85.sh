#!/usr/bin/env bash
set -euo pipefail

[[ $# -gt 0 ]] || { echo "Usage: $0 <command> [args...]" >&2; exit 2; }

install_root="${CANN85_INSTALL_ROOT:-${HOME}/ascend-envs/cann-8.5.0}"
venv_root="${CANN85_VENV:-${HOME}/venvs/xingchenbei-cann85}"

exec env -i \
    HOME="${HOME}" \
    USER="${USER:-$(id -un)}" \
    LOGNAME="${LOGNAME:-${USER:-$(id -un)}}" \
    SHELL="/bin/bash" \
    TERM="${TERM:-dumb}" \
    LANG="${LANG:-C.UTF-8}" \
    PATH="/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" \
    CANN85_INSTALL_ROOT="${install_root}" \
    CANN85_VENV="${venv_root}" \
    CANN85_EXPECTED_OPS_PACKAGE="${CANN85_EXPECTED_OPS_PACKAGE:-Ascend-cann-910b-ops}" \
    SPARSE_FLASH_ATTENTION_DEVICE_ID="${SPARSE_FLASH_ATTENTION_DEVICE_ID:-4}" \
    SPARSE_FLASH_ATTENTION_VALIDATE_TIMEOUT="${SPARSE_FLASH_ATTENTION_VALIDATE_TIMEOUT:-300}" \
    /bin/bash --noprofile --norc -c '
set -eo pipefail

mapfile -t toolkit_infos < <(find "${CANN85_INSTALL_ROOT}" -type f -name ascend_toolkit_install.info -print 2>/dev/null)
[[ ${#toolkit_infos[@]} -eq 1 ]] || { echo "Expected exactly one CANN Toolkit metadata file below ${CANN85_INSTALL_ROOT}; found ${#toolkit_infos[@]}" >&2; exit 2; }
toolkit_info=${toolkit_infos[0]}
version=$(sed -n "s/^version=//p" "${toolkit_info}" | head -1)
[[ "${version}" == "8.5.0" ]] || { echo "CANN version mismatch: ${version:-missing}" >&2; exit 2; }
cann_home=$(sed -n "s/^path=//p" "${toolkit_info}" | head -1)
if [[ -z "${cann_home}" || ! -d "${cann_home}" ]]; then
    cann_home=$(dirname "$(dirname "${toolkit_info}")")
fi
cann_home=$(readlink -f "${cann_home}")
install_root_real=$(readlink -f "${CANN85_INSTALL_ROOT}")
case "${cann_home}/" in
    "${install_root_real}/"*) ;;
    *) echo "CANN home escapes isolated install root: ${cann_home}" >&2; exit 2 ;;
esac
mapfile -t set_env_candidates < <(find "${cann_home}" -maxdepth 2 -type f -name set_env.sh -print)
[[ ${#set_env_candidates[@]} -eq 1 ]] || { echo "Expected exactly one set_env.sh below ${cann_home}" >&2; exit 2; }
set +u
# shellcheck disable=SC1090
source "${set_env_candidates[0]}"
set -u
export CANN85_HOME="${cann_home}"
export CANN_ENV_VERSION="${version}"
[[ -x "${CANN85_VENV}/bin/python" ]] || { echo "Missing isolated Python environment: ${CANN85_VENV}" >&2; exit 2; }
export PATH="${CANN85_VENV}/bin:${PATH}"
for var_name in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH; do
    var_value=${!var_name-}
    if grep -Eq "/usr/local/Ascend/(cann-9|ascend-toolkit|opp|compiler)(/|:|$)" <<<"${var_value}"; then
        echo "CANN 9 path leakage detected in ${var_name}" >&2
        exit 2
    fi
done
exec "$@"
' cann85-wrapper "$@"

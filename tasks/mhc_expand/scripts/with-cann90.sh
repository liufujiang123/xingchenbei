#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

cann_home="${CANN90_HOME:-/usr/local/Ascend/cann-9.0.0-beta.1}"
set_env="${cann_home}/set_env.sh"
if [[ ! -f "${set_env}" ]]; then
    echo "Missing CANN 9.0 environment script: ${set_env}" >&2
    exit 2
fi

exec env -i \
    HOME="${HOME}" \
    USER="${USER:-$(id -un)}" \
    LOGNAME="${LOGNAME:-${USER:-$(id -un)}}" \
    SHELL="/bin/bash" \
    TERM="${TERM:-dumb}" \
    LANG="${LANG:-C.UTF-8}" \
    PATH="/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" \
    CANN90_HOME="$(readlink -f "${cann_home}")" \
    CANN90_SET_ENV="$(readlink -f "${set_env}")" \
    MHC_EXPAND_DEVICE_ID="${MHC_EXPAND_DEVICE_ID:-0}" \
    /bin/bash --noprofile --norc -c '
set -eo pipefail
set +u
# shellcheck disable=SC1090
source "${CANN90_SET_ENV}"
set -u

for var_name in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME CANN_INSTALL_PATH; do
    var_value=${!var_name-}
    if grep -Fq "${HOME}/ascend-envs/cann-8.5.0" <<<"${var_value}"; then
        echo "CANN 8.5 path leakage detected in ${var_name}: ${var_value}" >&2
        exit 2
    fi
done

exec "$@"
' cann90-wrapper "$@"

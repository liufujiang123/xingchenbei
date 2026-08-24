#!/usr/bin/env bash
# Environment-only bootstrap for a fresh AtomGit Ascend 910B4 notebook/container.
# It DOES NOT build or run SparseFlashAttention.
#
# Recommended:
#   source tasks/sparse_flash_attention/scripts/setup_atomgit_910b_env.sh
#
# Optional overrides:
#   SFA_BRANCH=probe/sfa-910b-worker
#   SFA_WORK_ROOT="$HOME/sfa-910b"
#   SFA_CANN_HOME=/usr/local/Ascend/cann-8.5.0
#   SFA_GIT_RETRIES=6
#   SFA_GIT_ATTEMPT_TIMEOUT=90
#   SPARSE_FLASH_ATTENTION_DEVICE_ID=0

_sfa_env_main() {
    set -e

    local REPO_SLUG="${SFA_REPO_SLUG:-liufujiang123/xingchenbei}"
    local REPO_URL="${SFA_REPO_URL:-https://github.com/liufujiang123/xingchenbei.git}"
    local BRANCH="${SFA_BRANCH:-probe/sfa-910b-worker}"
    local WORK_ROOT="${SFA_WORK_ROOT:-$HOME/sfa-910b}"
    local REPO_DIR="${SFA_REPO_DIR:-$WORK_ROOT/repo}"
    local CANN_HOME="${SFA_CANN_HOME:-/usr/local/Ascend/cann-8.5.0}"
    local DEVICE_ID="${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}"
    local GIT_RETRIES="${SFA_GIT_RETRIES:-6}"
    local GIT_TIMEOUT="${SFA_GIT_ATTEMPT_TIMEOUT:-90}"
    local ENV_FILE="$HOME/.sfa910b_env.sh"
    local REPORT="$HOME/sfa910b-env-report.txt"
    local SET_ENV=""

    log()  { printf '\n\033[1;36m[SFA ENV]\033[0m %s\n' "$*"; }
    warn() { printf '\n\033[1;33m[SFA ENV WARN]\033[0m %s\n' "$*" >&2; }
    fail() { printf '\n\033[1;31m[SFA ENV FAIL]\033[0m %s\n' "$*" >&2; return 1; }
    have() { command -v "$1" >/dev/null 2>&1; }

    git_net() {
        local label="$1"
        shift
        local attempt delay
        for ((attempt=1; attempt<=GIT_RETRIES; attempt++)); do
            if timeout "$GIT_TIMEOUT" env GIT_TERMINAL_PROMPT=0 \
                git -c http.version=HTTP/1.1 \
                    -c http.lowSpeedLimit=1 \
                    -c http.lowSpeedTime=30 \
                    "$@"; then
                return 0
            fi
            if (( attempt == GIT_RETRIES )); then
                break
            fi
            delay=$((attempt * 8))
            warn "$label failed (attempt ${attempt}/${GIT_RETRIES}); retrying in ${delay}s"
            sleep "$delay"
        done
        fail "$label failed after ${GIT_RETRIES} attempts"
        return 1
    }

    clone_repo() {
        local parent tmp attempt delay
        parent="$(dirname "$REPO_DIR")"
        mkdir -p "$parent"
        tmp="${REPO_DIR}.clone-tmp"

        for ((attempt=1; attempt<=GIT_RETRIES; attempt++)); do
            rm -rf "$tmp"
            if timeout "$GIT_TIMEOUT" env GIT_TERMINAL_PROMPT=0 \
                git -c http.version=HTTP/1.1 \
                    -c http.lowSpeedLimit=1 \
                    -c http.lowSpeedTime=30 \
                    clone --depth 1 --single-branch --branch "$BRANCH" \
                    "$REPO_URL" "$tmp"; then
                if [[ -e "$REPO_DIR" && ! -d "$REPO_DIR/.git" ]]; then
                    rm -rf "$tmp"
                    fail "$REPO_DIR exists but is not a Git repository; refusing to overwrite it"
                    return 1
                fi
                rm -rf "$REPO_DIR"
                mv "$tmp" "$REPO_DIR"
                return 0
            fi
            rm -rf "$tmp"
            if (( attempt == GIT_RETRIES )); then
                break
            fi
            delay=$((attempt * 8))
            warn "git clone failed (attempt ${attempt}/${GIT_RETRIES}); retrying in ${delay}s"
            sleep "$delay"
        done
        fail "git clone failed after ${GIT_RETRIES} attempts"
        return 1
    }

    install_apt_packages() {
        local missing=("$@")
        ((${#missing[@]})) || return 0
        have sudo || { fail "Missing commands: ${missing[*]}, and sudo is unavailable."; return 1; }

        log "Installing missing packages: ${missing[*]}"
        if ! sudo apt-get install -y "${missing[@]}"; then
            log "Package index may be stale; running apt-get update once."
            sudo apt-get update
            sudo apt-get install -y "${missing[@]}"
        fi
    }

    log "1/7 Check/install basic tools"
    local pkgs=()
    have git   || pkgs+=(git)
    have gh    || pkgs+=(gh)
    have cmake || pkgs+=(cmake)
    have g++   || pkgs+=(build-essential)
    have make  || pkgs+=(build-essential)
    have curl  || pkgs+=(curl)
    install_apt_packages "${pkgs[@]}"

    have python3 || { fail "python3 is missing; this AtomGit image is not the expected runtime image."; return 1; }
    have npu-smi || { fail "npu-smi is missing; Ascend device/driver is not exposed in this environment."; return 1; }
    have timeout || { fail "timeout command is unavailable."; return 1; }

    log "2/7 Activate CANN 8.5.0"
    if [[ ! -d "$CANN_HOME" ]]; then
        CANN_HOME="$(find /usr/local/Ascend -maxdepth 3 -type d -name 'cann-8.5.0' -print -quit 2>/dev/null || true)"
    fi
    [[ -n "$CANN_HOME" && -d "$CANN_HOME" ]] || {
        fail "Cannot find CANN 8.5.0 under /usr/local/Ascend."
        return 1
    }
    CANN_HOME="$(readlink -f "$CANN_HOME")"

    for candidate in \
        "$CANN_HOME/set_env.sh" \
        "$CANN_HOME/ascend-toolkit/set_env.sh" \
        "$CANN_HOME/toolkit/set_env.sh"
    do
        if [[ -r "$candidate" ]]; then
            SET_ENV="$candidate"
            break
        fi
    done
    if [[ -z "$SET_ENV" ]]; then
        SET_ENV="$(find "$CANN_HOME" -maxdepth 4 -type f -name set_env.sh -print -quit 2>/dev/null || true)"
    fi
    [[ -n "$SET_ENV" && -r "$SET_ENV" ]] || {
        fail "CANN exists at $CANN_HOME, but set_env.sh was not found."
        return 1
    }

    # shellcheck disable=SC1090
    source "$SET_ENV"
    export ASCEND_HOME_PATH="$CANN_HOME"
    export SPARSE_FLASH_ATTENTION_DEVICE_ID="$DEVICE_ID"

    local VERSION_OK=0
    while IFS= read -r info; do
        if grep -qx 'version=8.5.0' "$info" 2>/dev/null; then
            VERSION_OK=1
            break
        fi
    done < <(find "$ASCEND_HOME_PATH" -maxdepth 4 -type f -name ascend_toolkit_install.info -print 2>/dev/null)
    [[ "$VERSION_OK" -eq 1 ]] || {
        fail "CANN directory found, but toolkit metadata does not verify version=8.5.0."
        return 1
    }

    cat > "$ENV_FILE" <<EOF
# Auto-generated by setup_atomgit_910b_env.sh
source "$SET_ENV"
export ASCEND_HOME_PATH="$CANN_HOME"
export SPARSE_FLASH_ATTENTION_DEVICE_ID="$DEVICE_ID"
EOF

    if ! grep -Fq 'source "$HOME/.sfa910b_env.sh"' "$HOME/.bashrc" 2>/dev/null; then
        {
            echo
            echo '# SFA 910B environment'
            echo '[[ -r "$HOME/.sfa910b_env.sh" ]] && source "$HOME/.sfa910b_env.sh"'
        } >> "$HOME/.bashrc"
    fi

    log "3/7 Configure GitHub CLI authentication"
    if ! gh auth status --hostname github.com >/dev/null 2>&1; then
        echo
        echo "GitHub authorization is the only interactive step."
        echo "When a one-time code appears, open this on your own computer:"
        echo "  https://github.com/login/device"
        echo
        gh auth login --hostname github.com --web
    fi

    gh auth setup-git
    git config --global http.version HTTP/1.1

    log "4/7 Verify GitHub repository access"
    gh auth status --hostname github.com || {
        fail "GitHub CLI authentication is not healthy."
        return 1
    }

    local CAN_PUSH=""
    local api_attempt
    for api_attempt in 1 2 3; do
        CAN_PUSH="$(gh api "repos/$REPO_SLUG" --jq '.permissions.push' 2>/dev/null || true)"
        [[ -n "$CAN_PUSH" ]] && break
        sleep $((api_attempt * 3))
    done
    if [[ "$CAN_PUSH" == "true" ]]; then
        echo "Repository write permission: PASS ($REPO_SLUG)"
    else
        warn "Could not verify .permissions.push=true for $REPO_SLUG."
    fi

    git_net "GitHub repository read test" ls-remote "$REPO_URL" HEAD >/dev/null
    echo "Repository read authentication: PASS"

    log "5/7 Clone/update working repository"
    if [[ -d "$REPO_DIR/.git" ]]; then
        if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
            warn "$REPO_DIR already exists and has local changes; leaving it untouched."
        else
            git_net "git fetch" -C "$REPO_DIR" fetch origin "$BRANCH"
            if git -C "$REPO_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
                git -C "$REPO_DIR" checkout "$BRANCH"
                git_net "git pull" -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
            else
                git -C "$REPO_DIR" checkout -b "$BRANCH" "origin/$BRANCH"
            fi
        fi
    elif [[ -e "$REPO_DIR" ]]; then
        fail "$REPO_DIR exists but is not a Git repository; move/remove it and rerun."
        return 1
    else
        clone_repo
    fi
    export SFA_REPO_ROOT="$REPO_DIR"

    log "6/7 Verify Ascend/Python/build environment only"
    echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
    echo "SPARSE_FLASH_ATTENTION_DEVICE_ID=$SPARSE_FLASH_ATTENTION_DEVICE_ID"
    python3 -V
    cmake --version | head -n 1
    g++ --version | head -n 1
    npu-smi info | sed -n '1,35p'

    python3 - <<'PY'
import ctypes
ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
print("libascendcl.so load: PASS")
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as exc:
    print("numpy import: WARN:", exc)
PY

    log "7/7 Write environment report"
    {
        echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "REPO=$REPO_SLUG"
        echo "BRANCH=$BRANCH"
        echo "REPO_DIR=$REPO_DIR"
        echo "COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
        echo "SET_ENV=$SET_ENV"
        echo "DEVICE_ID=$SPARSE_FLASH_ATTENTION_DEVICE_ID"
        echo "PYTHON=$(python3 -V 2>&1)"
        echo "CMAKE=$(cmake --version | head -n1)"
        echo "GXX=$(g++ --version | head -n1)"
        echo "GIT=$(git --version)"
        echo "GH=$(gh --version | head -n1)"
        echo "GIT_RETRIES=$GIT_RETRIES"
        echo "GIT_ATTEMPT_TIMEOUT=$GIT_TIMEOUT"
        echo
        echo "===== NPU ====="
        npu-smi info
        echo
        echo "===== GitHub ====="
        gh auth status --hostname github.com 2>&1 || true
        echo "repo_push_permission=${CAN_PUSH:-unknown}"
        echo
        echo "===== Safe selected environment ====="
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "PYTHONPATH=${PYTHONPATH:-}"
    } > "$REPORT" 2>&1

    echo
    echo "============================================================"
    echo "Environment setup complete."
    echo "Repo:   $REPO_DIR"
    echo "Branch: $BRANCH"
    echo "Report: $REPORT"
    echo
    echo "This script did NOT build or run SparseFlashAttention."
    echo "CANN activation has also been added to ~/.bashrc for this container."
    echo "============================================================"

    cd "$REPO_DIR"
}

_sfa_env_main "$@"
_sfa_rc=$?
unset -f _sfa_env_main 2>/dev/null || true
return "$_sfa_rc" 2>/dev/null || exit "$_sfa_rc"

#!/usr/bin/env bash
set -euo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
repo_root=$(cd "${task_dir}/../.." && pwd)
archive=${1:-}

fail() {
    echo "SFA_910B_RESULT_UPLOAD_FAIL: $*" >&2
    exit 1
}

[[ -n "${archive}" && -r "${archive}" ]] || fail "usage: $0 <result-archive>"
command -v git >/dev/null || fail "git is unavailable"
remote_url=$(git -C "${repo_root}" remote get-url origin) || fail "origin remote is unavailable"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_branch="probe/sfa-910b-result-${timestamp}"
result_repo=$(mktemp -d "${TMPDIR:-/tmp}/sfa-910b-result-push.XXXXXX")
cleanup() { rm -rf "${result_repo}"; }
trap cleanup EXIT

git init -q "${result_repo}"
git -C "${result_repo}" config user.name "SFA 910B Notebook Runner"
git -C "${result_repo}" config user.email "sfa-910b-notebook@users.noreply.github.com"
git -C "${result_repo}" remote add origin "${remote_url}"
cp "${archive}" "${result_repo}/sfa-910b-probe-results.tar.gz"
{
    echo "source_commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "created_utc=${timestamp}"
    echo "archive_sha256=$(sha256sum "${archive}" | awk '{print $1}')"
} >"${result_repo}/metadata.txt"
git -C "${result_repo}" add sfa-910b-probe-results.tar.gz metadata.txt
git -C "${result_repo}" commit -q -m "test: upload SFA 910B probe result ${timestamp}"

# AtomGit notebooks are configured with GitHub CLI's HTTPS credential helper.
# Do not probe SSH-over-443 here: fresh notebook environments normally have no
# SSH key/known_hosts state, which only produces a misleading host-key failure.
# Force HTTP/1.1 because this environment has shown intermittent HTTP/2/GnuTLS
# termination, and retry transient failures without re-running the probe.
push_ok=0
retry_count=${SFA_910B_RESULT_UPLOAD_RETRIES:-6}
push_timeout=${SFA_910B_RESULT_UPLOAD_TIMEOUT:-90}
retry_delay=${SFA_910B_RESULT_UPLOAD_RETRY_DELAY:-10}

for ((attempt=1; attempt<=retry_count; attempt++)); do
    if GIT_TERMINAL_PROMPT=0 timeout "${push_timeout}" \
        git -C "${result_repo}" -c http.version=HTTP/1.1 \
        push origin "HEAD:refs/heads/${result_branch}"; then
        push_ok=1
        break
    fi

    status=$?
    echo "SFA_910B_RESULT_UPLOAD_RETRY=${attempt}/${retry_count} status=${status}" >&2
    if (( attempt < retry_count )); then
        sleep "${retry_delay}"
    fi
done

[[ ${push_ok} -eq 1 ]] || \
    fail "GitHub HTTPS result push failed after ${retry_count} attempts"

echo "RESULT_PUSH_BRANCH=${result_branch}"
echo "RESULT_PUSH=PASS"

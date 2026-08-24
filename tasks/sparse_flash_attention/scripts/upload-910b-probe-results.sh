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

# Notebook clones normally use HTTPS, which can suffer transient GnuTLS
# termination. Prefer GitHub's SSH-over-443 endpoint when a deploy/user key is
# available; BatchMode guarantees that a missing key never prompts. HTTPS
# through gh's credential helper remains the portable fallback.
ssh_remote_url=
if [[ "${remote_url}" =~ ^https://github\.com/(.+)\.git$ ]]; then
    ssh_remote_url="ssh://git@ssh.github.com:443/${BASH_REMATCH[1]}.git"
    git -C "${result_repo}" remote add ssh_origin "${ssh_remote_url}"
fi
push_ok=0
if [[ -n "${ssh_remote_url}" ]]; then
    if GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
        git -C "${result_repo}" push ssh_origin "HEAD:refs/heads/${result_branch}"; then
        push_ok=1
    else
        echo "SFA_910B_RESULT_UPLOAD_SSH443_UNAVAILABLE=1" >&2
    fi
fi
for attempt in 1 2 3; do
    [[ ${push_ok} -eq 0 ]] || break
    if git -C "${result_repo}" -c http.version=HTTP/1.1 \
        push origin "HEAD:refs/heads/${result_branch}"; then
        push_ok=1
        break
    fi
    echo "SFA_910B_RESULT_UPLOAD_RETRY=${attempt}" >&2
done
[[ ${push_ok} -eq 1 ]] || fail "GitHub result push failed after three HTTP/1.1 attempts"

echo "RESULT_PUSH_BRANCH=${result_branch}"
echo "RESULT_PUSH=PASS"

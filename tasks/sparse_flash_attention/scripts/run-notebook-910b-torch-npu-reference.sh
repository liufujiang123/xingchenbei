#!/usr/bin/env bash
# Validate the installed official torch_npu SparseFlashAttention on a real
# 910B.  This runner deliberately does not build, install, or preload the
# competition custom operator.
set -uo pipefail

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "${script_path}")
task_dir=$(cd "${script_dir}/.." && pwd)
test_file="${task_dir}/tests/test_sparse_flash_attention_torch_npu_reference.py"
device_id=${SPARSE_FLASH_ATTENTION_DEVICE_ID:-0}

fail() {
    echo "SFA_910B_TORCH_NPU_REFERENCE_FAIL: $*" >&2
    exit 1
}

[[ "${device_id}" =~ ^[0-9]+$ ]] || fail "invalid device id: ${device_id}"
[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || \
    fail "activate CANN first; ASCEND_HOME_PATH is missing or invalid"
[[ -r "${test_file}" ]] || fail "missing runner: ${test_file}"
command -v python3 >/dev/null || fail "python3 is unavailable"
command -v npu-smi >/dev/null || fail "npu-smi is unavailable"

run_root=${SFA_910B_TORCH_NPU_REFERENCE_ROOT:-"${TMPDIR:-/tmp}/sfa-910b-torch-npu-reference"}
if [[ -e "${run_root}" ]]; then
    [[ -d "${run_root}" ]] || fail "result root is not a directory: ${run_root}"
    find "${run_root}" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    find "${run_root}" -type f -exec chmod u+rw {} + 2>/dev/null || true
    find "${run_root}" -depth -delete
fi
mkdir -p "${run_root}/logs" "${run_root}/results"

{
    echo "UTC_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "REPO_COMMIT=$(git -C "${task_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "ASCEND_HOME_PATH=$(readlink -f "${ASCEND_HOME_PATH}")"
    echo "DEVICE_ID=${device_id}"
    echo "PYTHON=$(python3 -V 2>&1)"
    npu-smi info
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 - <<'PY'
import torch
import torch_npu

print(f"TORCH={torch.__version__}")
print(f"TORCH_NPU={torch_npu.__version__}")
print(f"SFA_SCHEMA={torch.ops.npu.npu_sparse_flash_attention.default._schema}")
PY
} >"${run_root}/results/environment.txt" 2>&1

printf 'stage\tstatus\n' >"${run_root}/results/summary.tsv"
fingerprint_log="${run_root}/logs/fingerprint.log"
if timeout "${SFA_910B_REFERENCE_TIMEOUT:-180}" \
    python3 -u "${test_file}" fingerprint --device "${device_id}" \
    >"${fingerprint_log}" 2>&1; then
    fingerprint_status=VALUE_PASS
else
    status=$?
    if [[ ${status} -eq 124 ]]; then
        fingerprint_status=TIMEOUT
    elif [[ ${status} -eq 3 ]]; then
        fingerprint_status=NON_VALUE
    else
        fingerprint_status="FAIL_${status}"
    fi
fi
printf 'fingerprint\t%s\n' "${fingerprint_status}" >>"${run_root}/results/summary.tsv"
cp "${fingerprint_log}" "${run_root}/results/fingerprint.txt"

# The official runtime is never promoted to a reference after an ambiguous or
# KEY result.  In particular, do not let a broken upstream golden bless it.
if [[ "${fingerprint_status}" == VALUE_PASS ]]; then
    matrix_log="${run_root}/logs/reference-matrix.log"
    if timeout "${SFA_910B_REFERENCE_MATRIX_TIMEOUT:-900}" \
        python3 -u "${test_file}" all --device "${device_id}" \
        >"${matrix_log}" 2>&1; then
        matrix_status=PASS
    else
        status=$?
        if [[ ${status} -eq 124 ]]; then
            matrix_status=TIMEOUT
        else
            matrix_status="FAIL_${status}"
        fi
    fi
    printf 'reference_matrix\t%s\n' "${matrix_status}" >>"${run_root}/results/summary.tsv"
    cp "${matrix_log}" "${run_root}/results/reference-matrix.txt"
else
    printf 'reference_matrix\tREFUSED\n' >>"${run_root}/results/summary.tsv"
fi
echo "UTC_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${run_root}/results/environment.txt"
archive="${TMPDIR:-/tmp}/sfa-910b-torch-npu-reference-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "${archive}" -C "${run_root}" results logs

echo "===== SFA 910B TORCH_NPU REFERENCE SUMMARY ====="
column -t -s $'\t' "${run_root}/results/summary.tsv" 2>/dev/null || \
    command cat "${run_root}/results/summary.tsv"
echo "RESULT_ARCHIVE=${archive}"
if [[ "${SFA_910B_PUSH_RESULTS:-0}" == "1" ]]; then
    upload_script="${script_dir}/upload-910b-probe-results.sh"
    [[ -x "${upload_script}" ]] || fail "result upload helper is unavailable"
    "${upload_script}" "${archive}"
else
    echo "Set SFA_910B_PUSH_RESULTS=1 to upload the archive through the configured helper."
fi

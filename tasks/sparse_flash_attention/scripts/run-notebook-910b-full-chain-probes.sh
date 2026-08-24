#!/usr/bin/env bash
# Compare the complete pre-vector scalar kernel with the current vector kernel,
# then isolate 512-D content dot and value accumulation changes.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SFA_910B_DEVICE_API_VARIANTS="baseline old_scalar content_scalar value_scalar content_value_scalar"
export SFA_910B_DEVICE_API_CASES="L_rope_single_index L_content_single_index D_rope_required D_rope_required_fp32 A_basic L_random_diffuse"
exec "${script_dir}/run-notebook-910b-device-api-probes.sh"

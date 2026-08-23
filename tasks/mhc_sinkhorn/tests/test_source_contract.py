"""Static red-line tests for the package-facing MhcSinkhorn contract."""

from __future__ import annotations

from pathlib import Path


TASK = Path(__file__).resolve().parents[1]
CODE = TASK / "workspace" / "code"


def source(relative: str) -> str:
    return (CODE / relative).read_text(encoding="utf-8")


def test_public_tensor_topology_and_attrs_are_frozen():
    host = source("op_host/mhc_sinkhorn.cpp")
    logits = host.index('Input("logits")')
    mask = host.index('Input("mask")')
    weights = host.index('Output("weights")')
    iterations = host.index('Attr("iterations")')
    eps = host.index('Attr("eps")')
    assert logits < mask < weights < iterations < eps
    assert '.Int(DEFAULT_ITERATIONS)' in host
    assert '.Float(DEFAULT_EPS)' in host
    for forbidden in ('Input("x")', 'Output("output")', "normOut", "sumOut"):
        assert forbidden not in host


def test_dtype_shape_and_mask_specializations_are_present():
    host = source("op_host/mhc_sinkhorn.cpp")
    key = source("op_kernel/tiling_key_mhc_sinkhorn.h")
    assert "ge::DT_FLOAT16" in host and "ge::DT_FLOAT" in host
    assert "rowDim != 4 && rowDim != 6 && rowDim != 8" in host
    assert "MASK_MODE_SCALAR" in host and "MASK_MODE_FULL" in host
    assert "C_DT_FLOAT16, C_DT_FLOAT" in key
    assert "ASCENDC_TPL_UI_LIST, 4, 6, 8" in key
    assert "ASCENDC_TPL_UI_LIST, 0, 1, 2" in key


def test_kernel_gm_topology_fp32_state_and_stage_order_are_frozen():
    kernel = source("op_kernel/mhc_sinkhorn.cpp")
    signature = (
        "void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,\n"
        "                                        GM_ADDR workspace, GM_ADDR tiling)"
    )
    assert signature in kernel
    assert "LocalTensor<float> state" in kernel
    assert "for (uint32_t iteration = 1; iteration < iterations; ++iteration)" in kernel
    for suffix in ("N8", "Padded"):
        softmax = kernel.index(f"StableRowSoftmax{suffix}(state);")
        first_column = kernel.index(f"NormalizeColumns{suffix}(state);", softmax)
        remaining_row = kernel.index(f"NormalizeRows{suffix}(state);", first_column)
        remaining_column = kernel.index(
            f"NormalizeColumns{suffix}(state);", remaining_row
        )
        assert softmax < first_column < remaining_row < remaining_column


def test_official_source_remains_910b_only():
    cmake = source("CMakeLists.txt")
    host = source("op_host/mhc_sinkhorn.cpp")
    combined = cmake + host + source("op_kernel/mhc_sinkhorn.cpp")
    assert cmake.count("set(ASCEND_COMPUTE_UNIT ascend910b)") == 1
    assert host.count('.AddConfig("ascend910b");') == 1
    assert "ascend910_93" not in combined

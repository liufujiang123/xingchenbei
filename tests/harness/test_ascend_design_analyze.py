#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ascend_design_analyze


def make_tree(files):
    root = pathlib.Path(tempfile.mkdtemp(prefix="ascend-design-test-"))
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def ids(result):
    return {item["id"] for item in result["decisions"]}


def test_filename_does_not_define_semantics():
    tree = make_tree(
        {
            "TASK.md": "Contract text without implementation semantics.\n",
            "op_kernel/reduction_attention_kernel.cpp": "void Kernel() {}\n",
        }
    )
    result = ascend_design_analyze.analyze([tree], limit=30)
    assert "reduction" not in result["suggested_archetypes"]
    assert "attention" not in result["suggested_archetypes"]
    assert "reduction.ownership_and_passes" not in ids(result)


def test_source_signal_is_advisory_not_declared_contract():
    tree = make_tree(
        {
            "TASK.md": "The authoritative contract is intentionally minimal here.\n",
            "op_kernel/kernel.cpp": "void Kernel() { AscendC::ReduceSum(x, y); }\n",
        }
    )
    result = ascend_design_analyze.analyze([tree], limit=30)
    assert result["declared_archetypes"] == []
    assert "reduction" in result["suggested_archetypes"]
    assert "reduction.ownership_and_passes" in ids(result)
    assert "operator_archetype_not_declared" in " ".join(result["unknowns"])


def test_explicit_archetype_enables_family_decisions():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    result = ascend_design_analyze.analyze([tree], archetypes=["scan"], limit=30)
    assert result["declared_archetypes"] == ["scan"]
    assert "scan.state_locality" in ids(result)


def test_essential_design_decisions_are_always_present():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    result = ascend_design_analyze.analyze([tree], limit=30)
    required = {
        "contract.freeze_public_interface",
        "contract.domain_and_host_consistency",
        "semantics.stage_graph",
        "parallelism.axes_and_ownership",
        "layout.physical_contiguity",
        "tiling.host_runtime_plan",
        "tiling.full_tile_and_tail",
        "memory.lifetime_plan",
        "precision.compute_contract",
        "specialization.generic_fallback",
        "platform.target_separation",
        "validation.matrix_and_localization",
    }
    assert required <= ids(result)


def test_unknown_archetype_fails_closed():
    tree = make_tree({"TASK.md": "Contract.\n"})
    try:
        ascend_design_analyze.analyze([tree], archetypes=["magic_kernel"], limit=30)
    except ValueError as exc:
        assert "unknown design archetype" in str(exc)
    else:
        raise AssertionError("unknown archetype should fail closed")


def main():
    tests = [
        test_filename_does_not_define_semantics,
        test_source_signal_is_advisory_not_declared_contract,
        test_explicit_archetype_enables_family_decisions,
        test_essential_design_decisions_are_always_present,
        test_unknown_archetype_fails_closed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

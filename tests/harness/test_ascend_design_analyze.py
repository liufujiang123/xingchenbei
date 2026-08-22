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
    result = ascend_design_analyze.analyze([tree])
    assert "reduction" not in result["suggested_archetypes"]
    assert "attention" not in result["suggested_archetypes"]
    assert "reduction.ownership_and_passes" not in ids(result)


def test_source_signal_is_advisory_and_context_is_sparse():
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
    assert len(result["decisions"]) <= 3
    assert result["attention_budget"]["active_limit"] == 3
    assert all(isinstance(item, str) for item in result["deferred_decision_ids"])


def test_explicit_archetype_enables_family_decision_without_dumping_catalog():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    result = ascend_design_analyze.analyze([tree], archetypes=["scan"], limit=30)
    assert result["declared_archetypes"] == ["scan"]
    assert "scan.state_locality" in ids(result)
    assert len(result["decisions"]) <= 3


def test_risk_signal_triggers_at_most_one_deep_dive():
    tree = make_tree(
        {
            "TASK.md": "Contract.\n",
            "op_kernel/kernel.cpp": (
                "void Kernel() { AscendC::Matmul(a,b); Vector(x); "
                "CrossCoreSetFlag<0x0, PIPE_FIX>(1); workspace[0] = 0; }\n"
            ),
        }
    )
    result = ascend_design_analyze.analyze([tree], archetypes=["composite"], limit=30)
    deep = [item for item in result["decisions"] if item["detail_level"] == "deep_dive"]
    assert len(deep) <= 1
    assert any(item["id"] == "mixed_cv.protocol_first" for item in result["decisions"])
    if deep:
        assert "validate" in deep[0]
    assert all(("validate" in item) == (item["detail_level"] == "deep_dive") for item in result["decisions"])


def test_explicit_expansion_is_separate_from_active_budget():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    result = ascend_design_analyze.analyze([tree], limit=3, expand_ids=["memory.lifetime_plan"])
    assert len(result["decisions"]) <= 3
    assert [item["id"] for item in result["expanded_decisions"]] == ["memory.lifetime_plan"]
    assert result["expanded_decisions"][0]["detail_level"] == "deep_dive"
    assert "validate" in result["expanded_decisions"][0]


def test_unknown_archetype_and_pattern_fail_closed():
    tree = make_tree({"TASK.md": "Contract.\n"})
    try:
        ascend_design_analyze.analyze([tree], archetypes=["magic_kernel"])
    except ValueError as exc:
        assert "unknown design archetype" in str(exc)
    else:
        raise AssertionError("unknown archetype should fail closed")

    try:
        ascend_design_analyze.analyze([tree], expand_ids=["missing.pattern"])
    except ValueError as exc:
        assert "unknown design pattern" in str(exc)
    else:
        raise AssertionError("unknown design pattern should fail closed")


def main():
    tests = [
        test_filename_does_not_define_semantics,
        test_source_signal_is_advisory_and_context_is_sparse,
        test_explicit_archetype_enables_family_decision_without_dumping_catalog,
        test_risk_signal_triggers_at_most_one_deep_dive,
        test_explicit_expansion_is_separate_from_active_budget,
        test_unknown_archetype_and_pattern_fail_closed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
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


def make_registry():
    root = pathlib.Path(tempfile.mkdtemp(prefix="ascend-design-registry-"))
    path = root / "registry.json"
    patterns = [
        {
            "id": "contract.freeze",
            "title": "Freeze contract",
            "phase": "contract",
            "tier": "essential",
            "applies_to": ["all"],
            "when_tags": [],
            "decide": "Record the public contract and legal domain before implementation.",
            "guardrails": "Do not invent semantics.",
            "validate": "Compare with authoritative sources.",
        },
        {
            "id": "reduction.ownership",
            "title": "Reduction ownership",
            "phase": "architecture",
            "tier": "conditional",
            "applies_to": ["reduction"],
            "when_tags": ["reduction"],
            "decide": "Choose complete versus partial reduction ownership and merge strategy.",
            "guardrails": "Account for merge traffic and numerical order.",
            "validate": "Cover reduction boundaries and tails.",
        },
        {
            "id": "scan.state",
            "title": "Scan state ownership",
            "phase": "architecture",
            "tier": "conditional",
            "applies_to": ["scan"],
            "when_tags": ["scan"],
            "decide": "Keep carried state with its dependency chain when feasible.",
            "guardrails": "Do not split true recurrence for core count alone.",
            "validate": "Cover first, middle and final state transitions.",
        },
        {
            "id": "mixed.protocol",
            "title": "Mixed CV protocol",
            "phase": "architecture",
            "tier": "conditional",
            "applies_to": ["composite"],
            "when_tags": ["mixed_cv", "cross_core"],
            "decide": "Define producer, consumer, ready and reuse edges before synchronization primitives.",
            "guardrails": "Do not copy fixed credit or ring values.",
            "validate": "Check prologue, steady state and drain.",
        },
    ]
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "archetypes": ["reduction", "scan", "composite"],
                "patterns": patterns,
            }
        ),
        encoding="utf-8",
    )
    return path, patterns


def ids(items):
    return [item["id"] for item in items]


def test_catalog_exposes_every_summary_without_full_detail():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    registry, patterns = make_registry()
    result = ascend_design_analyze.analyze([tree], registry_path=registry)
    assert ids(result["experience_catalog"]) == [p["id"] for p in patterns]
    assert all("summary" in item for item in result["experience_catalog"])
    assert all("guardrails" not in item and "validate" not in item for item in result["experience_catalog"])
    assert result["decisions"] == []
    assert result["attention_budget"]["catalog_detail"] == "summary_only"


def test_machine_matching_is_advisory_not_automatic_selection():
    tree = make_tree(
        {
            "TASK.md": "Contract.\n",
            "op_kernel/kernel.cpp": "void Kernel() { AscendC::ReduceSum(x, y); }\n",
        }
    )
    registry, _ = make_registry()
    result = ascend_design_analyze.analyze([tree], registry_path=registry)
    assert "reduction" in result["suggested_archetypes"]
    assert "reduction.ownership" in ids(result["machine_suggestions"])
    assert result["selected_decisions"] == []
    assert result["decisions"] == []


def test_codex_can_select_pattern_not_chosen_by_machine():
    tree = make_tree({"TASK.md": "Contract.\n", "op_kernel/kernel.cpp": "void Kernel() {}\n"})
    registry, _ = make_registry()
    result = ascend_design_analyze.analyze(
        [tree],
        registry_path=registry,
        select_ids=["scan.state"],
    )
    assert ids(result["selected_decisions"]) == ["scan.state"]
    selected = result["selected_decisions"][0]
    assert selected["selected_by"] == "codex_selected"
    assert "decide" in selected and "guardrails" in selected and "validate" in selected


def test_selection_is_attention_bounded():
    tree = make_tree({"TASK.md": "Contract.\n"})
    registry, _ = make_registry()
    try:
        ascend_design_analyze.analyze(
            [tree],
            registry_path=registry,
            select_ids=["contract.freeze", "reduction.ownership", "scan.state", "mixed.protocol"],
        )
    except ValueError as exc:
        assert "select at most" in str(exc)
    else:
        raise AssertionError("selection must be bounded")


def test_filename_does_not_define_semantics():
    tree = make_tree(
        {
            "TASK.md": "Contract text without implementation semantics.\n",
            "op_kernel/reduction_scan_kernel.cpp": "void Kernel() {}\n",
        }
    )
    registry, _ = make_registry()
    result = ascend_design_analyze.analyze([tree], registry_path=registry)
    assert "reduction" not in result["suggested_archetypes"]
    assert "scan" not in result["suggested_archetypes"]


def test_unknown_archetype_and_pattern_fail_closed():
    tree = make_tree({"TASK.md": "Contract.\n"})
    registry, _ = make_registry()
    try:
        ascend_design_analyze.analyze([tree], registry_path=registry, archetypes=["magic_kernel"])
    except ValueError as exc:
        assert "unknown design archetype" in str(exc)
    else:
        raise AssertionError("unknown archetype should fail closed")

    try:
        ascend_design_analyze.analyze([tree], registry_path=registry, select_ids=["missing.pattern"])
    except ValueError as exc:
        assert "unknown design pattern" in str(exc)
    else:
        raise AssertionError("unknown pattern should fail closed")


def main():
    tests = [
        test_catalog_exposes_every_summary_without_full_detail,
        test_machine_matching_is_advisory_not_automatic_selection,
        test_codex_can_select_pattern_not_chosen_by_machine,
        test_selection_is_attention_bounded,
        test_filename_does_not_define_semantics,
        test_unknown_archetype_and_pattern_fail_closed,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

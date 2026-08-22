#!/usr/bin/env python3
"""Print compact research-derived Ascend optimization candidates."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "ascend_optimization_patterns.json"


def load_registry():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise SystemExit("unsupported optimization registry version")
    return data


def class_matches(pattern, operator_class):
    classes = pattern.get("classes", [])
    return "all" in classes or operator_class in classes


def select_patterns(registry, operator_class, bottlenecks, include_advanced):
    wanted = {x.lower() for x in bottlenecks}
    ranked = []
    for order, pattern in enumerate(registry["patterns"]):
        if not class_matches(pattern, operator_class):
            continue
        if pattern.get("tier", "core") == "advanced" and not include_advanced:
            continue
        tags = {str(x).lower() for x in pattern.get("tags", [])}
        overlap = len(wanted & tags)
        if wanted and overlap == 0:
            continue
        tier_penalty = 1 if pattern.get("tier") == "advanced" else 0
        ranked.append((-overlap, tier_penalty, order, pattern))
    return [item[3] for item in sorted(ranked)]


def main():
    parser = argparse.ArgumentParser(description="Ascend optimization candidate planner")
    parser.add_argument("--task")
    parser.add_argument("--operator-class", required=True, choices=["vector", "cube", "mixed_cv"])
    parser.add_argument("--bottleneck", action="append", default=[])
    parser.add_argument("--advanced", action="store_true", help="include advanced/SOC- or API-sensitive patterns")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    registry = load_registry()
    class_spec = registry["classes"][args.operator_class]
    patterns = select_patterns(registry, args.operator_class, args.bottleneck, args.advanced)
    patterns = patterns[: max(args.limit, 0)]

    payload = {
        "task": args.task,
        "operator_class": args.operator_class,
        "resource_model": class_spec["resource_model"],
        "canonical_flow": class_spec["canonical_flow"],
        "bottlenecks": args.bottleneck,
        "advanced": args.advanced,
        "candidates": patterns,
    }

    if args.as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.task:
        print("task=%s" % args.task)
    print("operator_class=%s" % args.operator_class)
    print("resource_model=%s" % " -> ".join(payload["resource_model"]))
    print("canonical_flow=%s" % payload["canonical_flow"])
    print("bottlenecks=%s" % (", ".join(args.bottleneck) if args.bottleneck else "unspecified"))
    print("advanced=%s\n" % ("yes" if args.advanced else "no"))

    if not patterns:
        print("No matching patterns. Re-check the bottleneck tags or omit --bottleneck.")
        return 0

    for index, pattern in enumerate(patterns, 1):
        print("[%d] %s — %s [%s]" % (index, pattern["id"], pattern["title"], pattern.get("tier", "core")))
        print("  when: %s" % pattern["when"])
        print("  try: %s" % pattern["try"])
        print("  avoid: %s" % pattern["avoid"])
        print("  evidence: %s" % ", ".join(pattern.get("evidence", [])))
        print()

    print("Rule: pick one major mechanism, state the expected Ascend resource effect, then build + correctness + same-case measurement decide keep/reject.")
    if not args.advanced:
        print("Advanced patterns are hidden by default; add --advanced only after profile/target-API evidence justifies them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

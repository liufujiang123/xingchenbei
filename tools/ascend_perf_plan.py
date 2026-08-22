#!/usr/bin/env python3
"""Print Ascend-first optimization candidates from the repository pattern registry."""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "ascend_optimization_patterns.json"


def load_registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def select_patterns(registry, operator_class, bottlenecks):
    spec = registry["operator_classes"][operator_class]
    patterns = list(registry.get("common", [])) + list(spec.get("patterns", []))
    wanted = {x.lower() for x in bottlenecks}
    if wanted:
        ranked = []
        for order, pattern in enumerate(patterns):
            tags = {str(x).lower() for x in pattern.get("bottlenecks", [])}
            overlap = len(wanted & tags)
            if overlap:
                ranked.append((-overlap, order, pattern))
        patterns = [x[2] for x in sorted(ranked)]
    return spec, patterns


def main():
    parser = argparse.ArgumentParser(description="Ascend optimization candidate planner")
    parser.add_argument("--task")
    parser.add_argument("--operator-class", required=True, choices=["vector", "cube", "mixed_cv"])
    parser.add_argument("--bottleneck", action="append", default=[])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    registry = load_registry()
    spec, patterns = select_patterns(registry, args.operator_class, args.bottleneck)
    patterns = patterns[: max(args.limit, 0)]
    payload = {
        "task": args.task,
        "operator_class": args.operator_class,
        "resource_model": spec.get("resource_model", []),
        "canonical_flow": spec.get("canonical_flow"),
        "bottlenecks": args.bottleneck,
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
    print("bottlenecks=%s\n" % (", ".join(args.bottleneck) if args.bottleneck else "unspecified"))
    for index, pattern in enumerate(patterns, 1):
        print("[%d] %s — %s" % (index, pattern.get("id"), pattern.get("title")))
        for label, key in (("signals", "signals"), ("candidate actions", "actions"), ("risks/gates", "risks")):
            print("  %s:" % label)
            for item in pattern.get(key, []):
                print("    - %s" % item)
        print()
    print("Rule: choose one major pattern per candidate; build + correctness + same-case performance decide keep/reject.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

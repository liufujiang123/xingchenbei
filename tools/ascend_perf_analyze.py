#!/usr/bin/env python3
"""Conservative Ascend performance diagnosis from source and profile evidence.

This module intentionally separates measured/observed profile symptoms from
static source risks. Static scanning generates hypotheses, never performance
claims.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "ascend_optimization_patterns.json"

CODE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".py"}
SKIP_DIRS = {
    ".git", ".agent-deps", "build", "output", "dist", "__pycache__",
    "runs", "profiles", "msprof_output", ".venv", "venv",
}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_PER_TAG = 6

CLASS_PATTERNS = {
    "cube": [
        r"\bMatmul\b", r"\bMmad\b", r"\bMMAD\b", r"\bTileMmad\b",
        r"\bT\.mma\b", r"\bT\.gemm", r"IS_ASCEND_AIC", r"\bPIPE_M\b",
        r"\bL0A\b", r"\bL0B\b", r"\bL0C\b", r"\bFixPipe\b",
    ],
    "vector": [
        r"IS_ASCEND_AIV", r"\bPIPE_V\b", r"\bT\.tile\.", r"\bT\.reduce_",
        r"\bAscendC::(?:Add|Sub|Mul|Div|Exp|Max|Min|Reduce|Cast|Compare|Select)\b",
        r"\bVector\b", r"\bLocalTensor\b",
    ],
}
MIXED_PATTERNS = [
    r"CrossCore(?:Set|Wait)Flag", r"\bT\.Scope\([\"']C[\"']\)",
    r"\bT\.Scope\([\"']V[\"']\)", r"KERNEL_TYPE_MIX", r"\bAIC\b.*\bAIV\b",
]

STATIC_RULES = [
    ("pipeline", [
        r"\bTPipe\b", r"\bTQue\b", r"\bT\.Pipelined\b", r"\bT\.pipelined\b",
        r"\bSetFlag\b", r"\bWaitFlag\b", r"\bPipeBarrier\b", r"num_stages",
        r"double.?buffer", r"ping.?pong", r"preload",
    ]),
    ("memory", [
        r"\bDataCopy(?:Pad)?\b", r"\bT\.copy\b", r"\bworkspace\b",
        r"\bGlobalTensor\b", r"\bLocalTensor\b", r"\balloc_(?:ub|L1|L0)",
        r"\bUB\b", r"\bL1\b", r"\bL0[ABC]\b",
    ]),
    ("bandwidth", [
        r"\bDataCopyPad\b", r"\bgather\b", r"\bpaged\b", r"\bsparse\b",
        r"\bworkspace\b", r"\bpadding\b", r"\bcontiguous\b",
    ]),
    ("cache", [
        r"\bresident\b", r"\bswizzle\b", r"\bring(?:Depth|_depth| depth)\b",
        r"\bL1\b", r"\bL2\b", r"\bgrouped\b",
    ]),
    ("scalar", [
        r"\bGetValue\b", r"\bSetValue\b", r"\bScalar\b",
        r"\bceil_div\b", r"\bCeilDiv\b", r"\bGetBlockIdx\b",
    ]),
    ("synchronization", [
        r"\bSetFlag\b", r"\bWaitFlag\b", r"\bPipeBarrier\b", r"\bSyncAll\b",
        r"CrossCore(?:Set|Wait)Flag", r"\bbarrier", r"\bsemaphore\b", r"\bcredit\b",
    ]),
    ("sparse", [
        r"\bsparse\b", r"\bpaged\b", r"\bgather\b", r"\bblock_table\b",
        r"\bindices\b", r"\bindex_", r"\bchunk_indices\b",
    ]),
    ("tiling", [
        r"\bTilingData\b", r"\btiling\b", r"\bblock_[MNK]\b",
        r"\bBLOCK_[MNK]\b", r"\bTileShape\b", r"\btile_shape\b",
    ]),
    ("underutilization", [
        r"\bGetBlockNum\b", r"\bGetBlockIdx\b", r"\bblockDim\b",
        r"\bcoreNum\b", r"\bcore_num\b", r"\bNUM_CORES\b",
    ]),
    ("compute", [
        r"\bMatmul\b", r"\bMmad\b", r"\bT\.mma\b", r"\bT\.gemm",
        r"\bT\.tile\.", r"\bReduce", r"\bExp\b",
    ]),
]

PROFILE_RULES = [
    ("pipeline", re.compile(r"(?:bubble|stall|idle|wait|gap).*(?:mte|vector|cube|pipe)|(?:mte|vector|cube|pipe).*(?:bubble|stall|idle|wait|gap)", re.I)),
    ("synchronization", re.compile(r"(?:sync|barrier|flag|semaphore|credit).*(?:stall|wait|overhead|bubble|idle)|(?:stall|wait|overhead|bubble|idle).*(?:sync|barrier|flag|semaphore|credit)", re.I)),
    ("underutilization", re.compile(r"(?:used.?core|core.?util|idle.?core|block.?util).*(?:low|idle|under|[0-5]\d(?:\.\d+)?%)", re.I)),
    ("cache", re.compile(r"(?:l1|l2|cache).*(?:miss|thrash|evict)|(?:miss|thrash|evict).*(?:l1|l2|cache)", re.I)),
    ("bandwidth", re.compile(r"(?:bandwidth|bw).*(?:low|bound|limit|util|stall)|(?:low|bound|limit|util|stall).*(?:bandwidth|bw)", re.I)),
    ("scalar", re.compile(r"scalar.*(?:bound|busy|stall|overhead|domin)|(?:bound|busy|stall|overhead|domin).*scalar", re.I)),
    ("memory", re.compile(r"(?:mte|memory|gm|dma).*(?:bound|stall|wait|bottleneck)|(?:bound|stall|wait|bottleneck).*(?:mte|memory|gm|dma)", re.I)),
    ("compute", re.compile(r"(?:vector|cube|aic|aiv).*(?:compute.?bound|bound|busy|utiliz)|(?:compute.?bound|bound|busy|utiliz).*(?:vector|cube|aic|aiv)", re.I)),
]

VALID_CLASSES = {"vector", "cube", "mixed_cv"}
VALID_TAGS = {
    "pipeline", "memory", "bandwidth", "cache", "compute", "latency",
    "underutilization", "scalar", "synchronization", "tiling", "sparse",
}


def _split_tokens(value):
    if not value:
        return []
    return [x for x in re.split(r"[\s,;:]+", value.strip()) if x]


def _rel(path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _iter_source_files(paths):
    seen = set()
    for root in paths:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > MAX_FILE_BYTES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _line_evidence(path, text, regex, tag):
    out = []
    compiled = re.compile(regex, re.I)
    for lineno, line in enumerate(text.splitlines(), 1):
        if compiled.search(line):
            out.append({
                "tag": tag,
                "file": _rel(path),
                "line": lineno,
                "excerpt": line.strip()[:220],
            })
            if len(out) >= MAX_EVIDENCE_PER_TAG:
                break
    return out


def scan_sources(paths):
    class_scores = {"vector": 0, "cube": 0}
    mixed_score = 0
    tag_scores = {}
    evidence = {}
    scanned = []

    class_compiled = {
        key: [(raw, re.compile(raw, re.I)) for raw in patterns]
        for key, patterns in CLASS_PATTERNS.items()
    }
    mixed_compiled = [(raw, re.compile(raw, re.I)) for raw in MIXED_PATTERNS]
    static_compiled = [
        (tag, [(raw, re.compile(raw, re.I)) for raw in patterns])
        for tag, patterns in STATIC_RULES
    ]

    for path in _iter_source_files(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned.append(_rel(path))

        for cls, rules in class_compiled.items():
            hits = sum(1 for _, regex in rules if regex.search(text))
            class_scores[cls] += hits

        mixed_score += sum(1 for _, regex in mixed_compiled if regex.search(text))

        for tag, rules in static_compiled:
            matched_rules = [raw for raw, regex in rules if regex.search(text)]
            if not matched_rules:
                continue
            tag_scores[tag] = tag_scores.get(tag, 0) + min(len(matched_rules), 3)
            bucket = evidence.setdefault(tag, [])
            if len(bucket) < MAX_EVIDENCE_PER_TAG:
                for raw in matched_rules:
                    for item in _line_evidence(path, text, raw, tag):
                        bucket.append(item)
                        break
                    if len(bucket) >= MAX_EVIDENCE_PER_TAG:
                        break

    static_tags = [
        tag for tag, score in sorted(tag_scores.items(), key=lambda item: (-item[1], item[0]))
        if score >= 2
    ]

    if class_scores["cube"] == 0 and class_scores["vector"] == 0:
        inferred_class = None
    elif class_scores["cube"] >= 2 and class_scores["vector"] >= 2 and mixed_score >= 1:
        inferred_class = "mixed_cv"
    elif class_scores["cube"] > class_scores["vector"]:
        inferred_class = "cube"
    else:
        inferred_class = "vector"

    return {
        "scanned_files": scanned,
        "class_scores": class_scores,
        "mixed_score": mixed_score,
        "inferred_class": inferred_class,
        "tag_scores": tag_scores,
        "static_risk_tags": static_tags,
        "evidence": evidence,
    }


def parse_profile(text):
    observed_tags = []
    evidence = []
    marker_class = None
    notes = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HARNESS_OPERATOR_CLASS="):
            value = line.split("=", 1)[1].strip()
            if value in VALID_CLASSES:
                marker_class = value
            continue
        if line.startswith("HARNESS_BOTTLENECKS="):
            for tag in _split_tokens(line.split("=", 1)[1]):
                if tag in VALID_TAGS and tag not in observed_tags:
                    observed_tags.append(tag)
                    evidence.append({"tag": tag, "source": "profile_marker", "excerpt": line[:240]})
            continue
        if line.startswith("HARNESS_PROFILE_NOTE="):
            notes.append(line.split("=", 1)[1].strip()[:500])
            continue

        for tag, regex in PROFILE_RULES:
            if regex.search(line) and tag not in observed_tags:
                observed_tags.append(tag)
                evidence.append({"tag": tag, "source": "profile_text", "excerpt": line[:240]})

    return {
        "operator_class": marker_class,
        "observed_tags": observed_tags,
        "evidence": evidence,
        "notes": notes,
    }


def load_registry():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise RuntimeError("unsupported optimization registry version")
    return data


def _class_matches(pattern, operator_class):
    classes = pattern.get("classes", [])
    return "all" in classes or operator_class in classes


def rank_patterns(registry, operator_class, observed_tags, hint_tags, static_tags, include_advanced=False, limit=5):
    weights = {}
    origins = {}
    for tag in static_tags:
        weights[tag] = max(weights.get(tag, 0), 1)
        origins.setdefault(tag, set()).add("static")
    for tag in hint_tags:
        weights[tag] = max(weights.get(tag, 0), 3)
        origins.setdefault(tag, set()).add("configured")
    for tag in observed_tags:
        weights[tag] = max(weights.get(tag, 0), 6)
        origins.setdefault(tag, set()).add("profile")

    ranked = []
    for order, pattern in enumerate(registry["patterns"]):
        if not _class_matches(pattern, operator_class):
            continue
        if pattern.get("tier", "core") == "advanced" and not include_advanced:
            continue
        matched = sorted(set(pattern.get("tags", [])) & set(weights))
        if not matched:
            continue
        score = sum(weights[tag] for tag in matched)
        profile_matches = sum(1 for tag in matched if "profile" in origins.get(tag, ()))
        ranked.append((-score, -profile_matches, order, pattern, matched))

    out = []
    for neg_score, _, _, pattern, matched in sorted(ranked)[:max(limit, 0)]:
        out.append({
            "id": pattern["id"],
            "title": pattern["title"],
            "tier": pattern.get("tier", "core"),
            "score": -neg_score,
            "matched_tags": matched,
            "when": pattern["when"],
            "try": pattern["try"],
            "avoid": pattern["avoid"],
            "evidence": pattern.get("evidence", []),
        })
    return out


def analyze(source_paths, profile_text="", operator_class="auto", bottleneck_hints=None,
            include_advanced=False, limit=5):
    bottleneck_hints = [x for x in (bottleneck_hints or []) if x in VALID_TAGS]
    source = scan_sources(source_paths)
    profile = parse_profile(profile_text)

    if operator_class and operator_class != "auto":
        resolved_class = operator_class
        class_source = "configured"
    elif profile["operator_class"]:
        resolved_class = profile["operator_class"]
        class_source = "profile_marker"
    elif source["inferred_class"]:
        resolved_class = source["inferred_class"]
        class_source = "static_source"
    else:
        resolved_class = None
        class_source = "unresolved"

    static_tags = source["static_risk_tags"]
    observed_tags = profile["observed_tags"]

    if observed_tags:
        planning_static = [tag for tag in static_tags if tag in observed_tags][:3]
    elif bottleneck_hints:
        planning_static = [tag for tag in static_tags if tag in bottleneck_hints]
        if not planning_static:
            planning_static = static_tags[:2]
    else:
        planning_static = static_tags[:4]

    registry = load_registry()
    candidates = []
    if resolved_class is not None:
        candidates = rank_patterns(
            registry,
            resolved_class,
            observed_tags,
            bottleneck_hints,
            planning_static,
            include_advanced=include_advanced,
            limit=limit,
        )

    conflicts = []
    if profile["operator_class"] and source["inferred_class"] and profile["operator_class"] != source["inferred_class"]:
        conflicts.append(
            "profile marker class %s differs from static source class %s"
            % (profile["operator_class"], source["inferred_class"])
        )

    if observed_tags:
        confidence = "profile_observed"
    elif bottleneck_hints:
        confidence = "configured_hypothesis"
    elif static_tags:
        confidence = "static_hypothesis"
    else:
        confidence = "low"

    return {
        "operator_class": resolved_class,
        "operator_class_source": class_source,
        "confidence": confidence,
        "observed_bottlenecks": observed_tags,
        "configured_hints": bottleneck_hints,
        "static_risk_tags": static_tags,
        "planning_tags": sorted(set(observed_tags + bottleneck_hints + planning_static)),
        "profile": profile,
        "source": source,
        "conflicts": conflicts,
        "candidates": candidates,
        "rule": "profile evidence outranks configured hints; static source signals are hypotheses only",
    }


def _resolve_default_sources(task):
    task_dir = ROOT / "tasks" / task
    candidates = [task_dir / "workspace", task_dir]
    out = []
    seen = set()
    for path in candidates:
        if path.exists():
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(path)
    return out


def main():
    parser = argparse.ArgumentParser(description="Ascend source/profile bottleneck diagnosis")
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-dir", action="append", default=[])
    parser.add_argument("--profile-file", action="append", default=[])
    parser.add_argument("--operator-class", default="auto", choices=["auto", "vector", "cube", "mixed_cv"])
    parser.add_argument("--bottleneck-hint", action="append", default=[])
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    sources = [pathlib.Path(x) for x in args.source_dir] or _resolve_default_sources(args.task)
    profile_chunks = []
    for raw in args.profile_file:
        path = pathlib.Path(raw)
        profile_chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    result = analyze(
        sources,
        profile_text="\n".join(profile_chunks),
        operator_class=args.operator_class,
        bottleneck_hints=args.bottleneck_hint,
        include_advanced=args.advanced,
        limit=args.limit,
    )
    result["task"] = args.task
    result["source_dirs"] = [_rel(path) for path in sources]

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(rendered, encoding="utf-8")

    if args.as_json or args.output:
        sys.stdout.write(rendered)
        return 0

    print("task=%s" % args.task)
    print("operator_class=%s source=%s" % (result["operator_class"] or "unresolved", result["operator_class_source"]))
    print("confidence=%s" % result["confidence"])
    print("observed_bottlenecks=%s" % (",".join(result["observed_bottlenecks"]) or "none"))
    print("static_risk_tags=%s" % (",".join(result["static_risk_tags"]) or "none"))
    print("planning_tags=%s" % (",".join(result["planning_tags"]) or "none"))
    if result["conflicts"]:
        print("conflicts=%s" % " | ".join(result["conflicts"]))
    print()
    for index, candidate in enumerate(result["candidates"], 1):
        print("[%d] %s — %s" % (index, candidate["id"], candidate["title"]))
        print("  matched_tags: %s" % ", ".join(candidate["matched_tags"]))
        print("  try: %s" % candidate["try"])
        print("  avoid: %s" % candidate["avoid"])
    print()
    print("Static source tags are hypotheses. Do not report them as measured bottlenecks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

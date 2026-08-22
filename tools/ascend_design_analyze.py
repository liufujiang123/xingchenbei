#!/usr/bin/env python3
"""Conservative, attention-budgeted Ascend operator design analyzer.

The knowledge base may be broad, but only a tiny active subset is injected into
Codex context. Static source/document signals are advisory and never become
contract facts. Concrete algorithms, tile sizes, core counts, queue depths and
specialization thresholds remain Codex decisions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "ascend_design_patterns.json"

TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".py", ".md", ".json", ".txt"}
IGNORE_PARTS = {".git", ".agent-deps", "build", "output", "dist", "runs", "profiles", "__pycache__", "generated"}
MAX_FILE_BYTES = 768 * 1024
MAX_TOTAL_BYTES = 6 * 1024 * 1024
MAX_ACTIVE_PATTERNS = 3
MAX_DEEP_DIVES = 1
MAX_DEFERRED_IDS = 8
RISK_SIGNAL_TAGS = {
    "mode", "optional", "rank_dispatch", "host_materialize", "pad", "reduction",
    "scan", "recurrent", "state", "workspace", "sparse", "gather", "paged",
    "index", "cross_core", "mixed_cv",
}

SIGNAL_RULES = {
    "mode": [r"\bTilingKey\b", r"\btiling[_ ]?key\b", r"\bbackward\b", r"\bforward\b"],
    "optional": [r"\bGetOptionalInput\b", r"\bOptional\b", r"\bnullptr\b", r"\boptional input\b"],
    "rank_dispatch": [r"\bGetDimNum\b", r"\brank\b", r"\bdim(?:ension)? count\b"],
    "varlen": [r"\bcu_seqlens\b", r"\bactual[_ ]?seq", r"\bvarlen\b", r"\bvariable[- ]length\b"],
    "chunk": [r"\bchunk_indices\b", r"\bchunkSize\b", r"\bchunk_size\b", r"\bchunkIdx\b"],
    "sequence": [r"\bseqLen\b", r"\bseq_len\b", r"\bsequence length\b"],
    "workspace": [r"\bworkspace\b", r"\bWorkspace\b"],
    "host_materialize": [r"\.contiguous\s*\(", r"\bF\.pad\s*\(", r"\btorch\.pad\s*\(", r"\.cpu\s*\("],
    "pad": [r"\bDataCopyPad\b", r"\bpad_value\b", r"\bpadding\b"],
    "transpose": [r"\bTranspose\b", r"\btranspose\s*\(", r"\bmovedim\s*\("],
    "contiguous": [r"\.contiguous\s*\("],
    "vector": [r"\bPIPE_V\b", r"\bIS_ASCEND_AIV\b", r"\bT\.tile\.", r"\bVector\b", r"\bRegTensor\b"],
    "cube": [r"\bIS_ASCEND_AIC\b", r"\bMatmul\b", r"\bMatMul\b", r"\bMmad\b", r"\bMMAD\b", r"\bGemm\b", r"\bPIPE_M\b"],
    "cross_core": [r"\bCrossCore", r"\bSYNC_AIC_AIV\b", r"\bSYNC_AIV_AIC\b", r"\bcross[-_ ]core\b"],
    "reduction": [r"\bReduceSum\b", r"\bReduceMax\b", r"\bWholeReduce", r"\breduce_sum\b", r"\breduce_max\b"],
    "multi_pass": [r"\bPass\s*[123]\b", r"\bpass[_ ]?[123]\b", r"\bonline softmax\b"],
    "softmax": [r"\bsoftmax\b", r"\bSoftmax\b"],
    "normalization": [r"\bRMSNorm\b", r"\bLayerNorm\b", r"\bGroupNorm\b", r"\bnormalization\b"],
    "attention": [r"\bFlashAttention\b", r"\bflash_attention\b", r"\battention\b"],
    "scan": [r"\bcumsum\b", r"\bcummin\b", r"\bcummax\b", r"\bprefix[-_ ]?scan\b", r"\bscan axis\b"],
    "recurrent": [r"\brecurrent\b", r"\brecurrence\b", r"\bcarried state\b", r"\brunning state\b"],
    "state": [r"\bdhState\b", r"\bstate carry\b", r"\bpersistent state\b", r"\bstate workspace\b"],
    "sparse": [r"\bsparse\b", r"\bSparse\b"],
    "gather": [r"\bGather\b", r"\bgather\b"],
    "paged": [r"\bblock_table\b", r"\bpage table\b", r"\bpaged\b"],
    "index": [r"\bsparse_index\b", r"\bchunk_indices\b", r"\bblock_table\b"],
    "broadcast": [r"\bBroadcast\b", r"\bbroadcast\s*\(", r"\bbroadcast_to\b"],
    "elementwise": [r"\belementwise\b", r"\belement-wise\b"],
    "mask": [r"\bmask\b", r"\bMaskReg\b"],
    "compare": [r"\bCompare\b", r"\bcompare\s*\("],
    "tiling": [r"\bTilingData\b", r"\btilingData\b", r"\bSetBlockDim\b", r"\bGetBlockNum\b"],
}

ARCHETYPE_FROM_TAGS = {
    "broadcast": {"broadcast"}, "reduction": {"reduction"}, "scan": {"scan"},
    "recurrent": {"recurrent"}, "sparse": {"sparse", "paged"}, "gather": {"gather"},
    "matmul": {"cube"}, "normalization": {"softmax", "normalization"},
    "attention": {"attention"}, "elementwise": {"elementwise"}, "composite": {"cross_core"},
}

PHASE_ORDER = {
    "contract": 0, "semantics": 1, "parallelism": 2, "layout": 3, "tiling": 4,
    "memory": 5, "precision": 6, "architecture": 7, "implementation": 8,
    "platform": 9, "validation": 10,
}


def load_registry(path=REGISTRY):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("unsupported design registry version")
    return data


def is_ignored(path):
    return any(part in IGNORE_PARTS for part in path.parts)


def iter_text_files(paths):
    seen = set()
    total = 0
    for base in paths:
        base = pathlib.Path(base)
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else base.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or is_ignored(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                continue
            total += size
            yield path


def display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def first_match(text, regexes):
    for regex in regexes:
        match = re.search(regex, text, re.IGNORECASE | re.MULTILINE)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            return line, text.splitlines()[line - 1].strip()[:180]
    return None


def scan_sources(paths):
    signals = {}
    files = []
    task_contract_files = []
    interface_evidence = []
    host_sources = 0
    kernel_sources = 0
    for path in iter_text_files(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = display_path(path)
        files.append(rel)
        parts = set(path.parts)
        if path.name == "TASK.md":
            task_contract_files.append(rel)
        if "op_host" in parts:
            host_sources += 1
        if "op_kernel" in parts:
            kernel_sources += 1
        for regex in (r"\bOpDef\b", r"\.Input\s*\(", r"\.Output\s*\(", r"\.Attr\s*\("):
            match = re.search(regex, text)
            if match:
                interface_evidence.append({"file": rel, "line": text.count("\n", 0, match.start()) + 1, "kind": regex})
                if len(interface_evidence) >= 8:
                    break
        for tag, regexes in SIGNAL_RULES.items():
            if tag in signals and len(signals[tag]) >= 3:
                continue
            hit = first_match(text, regexes)
            if hit:
                line, snippet = hit
                signals.setdefault(tag, []).append({"file": rel, "line": line, "snippet": snippet})
    if "cube" in signals and "vector" in signals:
        signals.setdefault("mixed_cv", []).append({"file": "<derived>", "line": 0, "snippet": "both Cube/AIC and Vector/AIV source signals are present"})
    return {
        "files_scanned": len(files), "task_contract_files": task_contract_files,
        "host_source_count": host_sources, "kernel_source_count": kernel_sources,
        "public_interface_evidence": interface_evidence[:8], "signals": signals,
    }


def normalize_archetypes(values, registry):
    allowed = set(registry["archetypes"])
    out = []
    for value in values or []:
        for item in re.split(r"[\s,;:]+", value.strip()):
            if not item:
                continue
            item = item.lower()
            if item not in allowed:
                raise ValueError("unknown design archetype %r; allowed=%s" % (item, ",".join(sorted(allowed))))
            if item not in out:
                out.append(item)
    return out


def suggest_archetypes(signal_tags):
    tags = set(signal_tags)
    return [name for name, required in ARCHETYPE_FROM_TAGS.items() if tags & required]


def rank_patterns(registry, declared, suggested, signal_tags):
    tags = set(signal_tags)
    ranked = []
    for order, pattern in enumerate(registry["patterns"]):
        essential = pattern.get("tier") == "essential"
        applies = set(pattern.get("applies_to", []))
        when_tags = set(pattern.get("when_tags", []))
        declared_match = set(declared) & applies
        suggested_match = set(suggested) & applies
        matched_tags = tags & when_tags
        if declared_match and matched_tags:
            relevance = 90 + 5 * len(matched_tags)
            reasons = ["declared_archetype:" + ",".join(sorted(declared_match)), "signal:" + ",".join(sorted(matched_tags))]
        elif declared_match:
            relevance = 80
            reasons = ["declared_archetype:" + ",".join(sorted(declared_match))]
        elif suggested_match and matched_tags:
            relevance = 75 + 5 * len(matched_tags)
            reasons = ["static_archetype:" + ",".join(sorted(suggested_match)), "signal:" + ",".join(sorted(matched_tags))]
        elif "all" in applies and matched_tags:
            relevance = 70 + 5 * len(matched_tags)
            reasons = ["signal:" + ",".join(sorted(matched_tags))]
        elif essential:
            relevance = 40
            reasons = ["essential"]
        elif "all" in applies and not when_tags:
            relevance = 25
            reasons = ["general"]
        else:
            continue
        ranked.append({**pattern, "relevance": relevance, "reasons": reasons, "matched_signal_tags": sorted(matched_tags), "_order": order})
    ranked.sort(key=lambda item: (-item["relevance"], PHASE_ORDER.get(item.get("phase"), 99), item["_order"]))
    return ranked


def choose_active(ranked, limit):
    limit = min(max(int(limit), 0), MAX_ACTIVE_PATTERNS)
    active = []
    selected_ids = set()
    used_phases = set()
    for item in ranked:
        phase = item.get("phase")
        if phase in used_phases:
            continue
        active.append(item)
        selected_ids.add(item["id"])
        used_phases.add(phase)
        if len(active) >= limit:
            return active
    for item in ranked:
        if item["id"] in selected_ids:
            continue
        active.append(item)
        if len(active) >= limit:
            break
    return active


def compact_pattern(item, deep=False):
    out = {
        "id": item["id"], "title": item["title"], "phase": item.get("phase"),
        "tier": item.get("tier"), "relevance": item.get("relevance"),
        "reasons": item.get("reasons", []), "detail_level": "deep_dive" if deep else "prompt",
        "decide": item["decide"], "guardrails": item["guardrails"],
    }
    if deep:
        out["validate"] = item["validate"]
        out["deep_dive_trigger"] = item.get("matched_signal_tags", [])
    return out


def resolve_expansions(ranked, registry, expand_ids):
    requested = []
    for value in expand_ids or []:
        for item in re.split(r"[\s,;]+", value.strip()):
            if item and item not in requested:
                requested.append(item)
    if not requested:
        return []
    by_id = {item["id"]: item for item in ranked}
    registry_by_id = {item["id"]: item for item in registry["patterns"]}
    unknown = [item for item in requested if item not in registry_by_id]
    if unknown:
        raise ValueError("unknown design pattern(s): %s" % ",".join(unknown))
    expanded = []
    for pattern_id in requested:
        item = by_id.get(pattern_id)
        if item is None:
            item = {**registry_by_id[pattern_id], "relevance": 0, "reasons": ["explicit_expand"], "matched_signal_tags": []}
        expanded.append(compact_pattern(item, deep=True))
    return expanded


def analyze(paths, archetypes=None, limit=MAX_ACTIVE_PATTERNS, registry_path=REGISTRY, expand_ids=None, deep_dive_limit=MAX_DEEP_DIVES):
    registry = load_registry(registry_path)
    declared = normalize_archetypes(archetypes or [], registry)
    scan = scan_sources(paths)
    signal_tags = sorted(scan["signals"])
    suggested = suggest_archetypes(signal_tags)
    ranked = rank_patterns(registry, declared, suggested, signal_tags)
    active_raw = choose_active(ranked, limit)
    risk_candidates = [item for item in active_raw if set(item.get("matched_signal_tags", [])) & RISK_SIGNAL_TAGS]
    deep_ids = {item["id"] for item in risk_candidates[:max(0, min(int(deep_dive_limit), MAX_DEEP_DIVES))]}
    active = [compact_pattern(item, deep=item["id"] in deep_ids) for item in active_raw]
    active_ids = {item["id"] for item in active_raw}
    deferred = [item["id"] for item in ranked if item["id"] not in active_ids][:MAX_DEFERRED_IDS]
    unknowns = []
    if not scan["task_contract_files"]:
        unknowns.append("task_contract_not_found_in_scanned_paths")
    if not scan["public_interface_evidence"]:
        unknowns.append("public_interface_not_detected_from_source")
    if not declared:
        unknowns.append("operator_archetype_not_declared; Codex must resolve it from the contract")
    if not scan["kernel_source_count"]:
        unknowns.append("kernel_source_not_detected; implementation may not exist yet")
    return {
        "kind": "ascend_operator_design", "status": "advisory",
        "declared_archetypes": declared, "suggested_archetypes": suggested,
        "archetype_rule": "declared archetypes are hints; static suggestions are not contract facts and may be overridden after reading the task statement",
        "contract_evidence": {
            "task_contract_files": scan["task_contract_files"],
            "public_interface_evidence": scan["public_interface_evidence"],
            "host_source_count": scan["host_source_count"], "kernel_source_count": scan["kernel_source_count"],
        },
        "static_signals": scan["signals"], "files_scanned": scan["files_scanned"], "unknowns": unknowns,
        "attention_budget": {
            "active_limit": min(max(int(limit), 0), MAX_ACTIVE_PATTERNS),
            "deep_dive_limit": min(max(int(deep_dive_limit), 0), MAX_DEEP_DIVES),
            "policy": "top-k dynamic retrieval; one risk-triggered deep dive; deferred patterns remain ids only",
        },
        "decisions": active, "deferred_decision_ids": deferred,
        "expanded_decisions": resolve_expansions(ranked, registry, expand_ids),
        "rule": "Only the small active set should enter normal Codex context. Use deferred ids as navigation, not checklist obligations. Expand one pattern only when a concrete risk/unknown requires it. Codex owns the concrete algorithm, task mapping, tiling, buffer sizes and specialization choices; authoritative contract/build/correctness evidence overrides all heuristics.",
    }


def print_report(result):
    print("kind=%s status=%s" % (result["kind"], result["status"]))
    print("declared_archetypes=%s" % (",".join(result["declared_archetypes"]) or "none"))
    print("suggested_archetypes=%s" % (",".join(result["suggested_archetypes"]) or "none"))
    print("files_scanned=%d" % result["files_scanned"])
    print("static_signal_tags=%s" % (",".join(sorted(result["static_signals"])) or "none"))
    print("unknowns=%s" % ("; ".join(result["unknowns"]) or "none"))
    budget = result["attention_budget"]
    print("attention_budget=active:%d deep:%d" % (budget["active_limit"], budget["deep_dive_limit"]))
    print()
    for index, item in enumerate(result["decisions"], 1):
        print("[%d] %s — %s [%s]" % (index, item["id"], item["title"], item["detail_level"]))
        print("  decide: %s" % item["decide"])
        print("  guardrails: %s" % item["guardrails"])
        if item.get("validate"):
            print("  validate: %s" % item["validate"])
            print("  deep_dive_trigger: %s" % (",".join(item.get("deep_dive_trigger", [])) or "explicit"))
        print("  selected_by: %s" % "; ".join(item.get("reasons", [])))
        print()
    if result["deferred_decision_ids"]:
        print("deferred_ids=%s" % ",".join(result["deferred_decision_ids"]))
    if result["expanded_decisions"]:
        print("\nEXPLICIT EXPANSION")
        for item in result["expanded_decisions"]:
            print("%s — %s" % (item["id"], item["title"]))
            print("  decide: %s" % item["decide"])
            print("  guardrails: %s" % item["guardrails"])
            print("  validate: %s" % item["validate"])
    print("rule=%s" % result["rule"])


def main():
    parser = argparse.ArgumentParser(description="Ascend operator-development design analyzer")
    parser.add_argument("--task")
    parser.add_argument("--task-dir")
    parser.add_argument("--workspace")
    parser.add_argument("--source-dir", action="append", default=[])
    parser.add_argument("--archetype", action="append", default=[])
    parser.add_argument("--limit", type=int, default=MAX_ACTIVE_PATTERNS, help="active prompt count; hard-capped at %d" % MAX_ACTIVE_PATTERNS)
    parser.add_argument("--expand-pattern", action="append", default=[], help="explicitly expand one design pattern by id; may be repeated")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    paths = []
    if args.source_dir:
        paths.extend(pathlib.Path(x).expanduser() for x in args.source_dir)
    else:
        if args.task_dir:
            paths.append(pathlib.Path(args.task_dir).expanduser())
        elif args.task:
            paths.append(ROOT / "tasks" / args.task)
        if args.workspace:
            paths.append(pathlib.Path(args.workspace).expanduser())
        elif args.task:
            workspace = ROOT / "tasks" / args.task / "workspace"
            if workspace.exists():
                paths.append(workspace)
    if not paths:
        raise SystemExit("provide --task, --task-dir/--workspace, or --source-dir")
    resolved = [p if p.is_absolute() else ROOT / p for p in paths]
    result = analyze(resolved, archetypes=args.archetype, limit=args.limit, expand_ids=args.expand_pattern)
    result["source_dirs"] = [display_path(p) for p in resolved]
    if args.as_json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

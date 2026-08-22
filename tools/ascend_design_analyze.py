#!/usr/bin/env python3
"""Conservative Ascend operator-development design analyzer.

This tool does not generate kernel code and does not infer contract semantics from
file names. It turns explicit archetype hints plus strong source/document signals
into a compact design checklist for Codex.
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
IGNORE_PARTS = {
    ".git",
    ".agent-deps",
    "build",
    "output",
    "dist",
    "runs",
    "profiles",
    "__pycache__",
    "generated",
}
MAX_FILE_BYTES = 768 * 1024
MAX_TOTAL_BYTES = 6 * 1024 * 1024

# Strong content signals only. File names are intentionally not used to infer semantics.
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
    "broadcast": {"broadcast"},
    "reduction": {"reduction"},
    "scan": {"scan"},
    "recurrent": {"recurrent"},
    "sparse": {"sparse", "paged"},
    "gather": {"gather"},
    "matmul": {"cube"},
    "normalization": {"softmax", "normalization"},
    "attention": {"attention"},
    "elementwise": {"elementwise"},
    "composite": {"cross_core"},
}

PHASE_ORDER = {
    "contract": 0,
    "semantics": 1,
    "parallelism": 2,
    "layout": 3,
    "tiling": 4,
    "memory": 5,
    "precision": 6,
    "architecture": 7,
    "implementation": 8,
    "platform": 9,
    "validation": 10,
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
            snippet = text.splitlines()[line - 1].strip()
            return line, snippet[:180]
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
                line = text.count("\n", 0, match.start()) + 1
                interface_evidence.append({"file": rel, "line": line, "kind": regex})
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
        signals.setdefault("mixed_cv", []).append(
            {"file": "<derived>", "line": 0, "snippet": "both Cube/AIC and Vector/AIV source signals are present"}
        )

    return {
        "files_scanned": len(files),
        "task_contract_files": task_contract_files,
        "host_source_count": host_sources,
        "kernel_source_count": kernel_sources,
        "public_interface_evidence": interface_evidence[:8],
        "signals": signals,
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
    out = []
    for archetype, required in ARCHETYPE_FROM_TAGS.items():
        if tags & required:
            out.append(archetype)
    return out


def select_patterns(registry, declared, suggested, signal_tags, limit):
    tags = set(signal_tags)
    selected = []

    for order, pattern in enumerate(registry["patterns"]):
        essential = pattern.get("tier") == "essential"
        applies = set(pattern.get("applies_to", []))
        when_tags = set(pattern.get("when_tags", []))
        declared_match = set(declared) & applies
        suggested_match = set(suggested) & applies
        tag_match = len(tags & when_tags)
        if essential:
            relevance = 100
            reasons = ["essential"]
        elif declared_match:
            relevance = 35 + 4 * tag_match
            reasons = ["declared_archetype:" + ",".join(sorted(declared_match))]
            if tags & when_tags:
                reasons.append("signal:" + ",".join(sorted(tags & when_tags)))
        elif suggested_match and tag_match:
            relevance = 25 + 4 * tag_match
            reasons = [
                "static_archetype:" + ",".join(sorted(suggested_match)),
                "signal:" + ",".join(sorted(tags & when_tags)),
            ]
        elif "all" in applies and tag_match:
            relevance = 20 + 4 * tag_match
            reasons = ["signal:" + ",".join(sorted(tags & when_tags))]
        elif "all" in applies and not when_tags:
            relevance = 15
            reasons = ["general"]
        else:
            continue
        selected.append(
            (
                -relevance,
                PHASE_ORDER.get(pattern.get("phase"), 99),
                order,
                {
                    **pattern,
                    "relevance": relevance,
                    "reasons": reasons,
                },
            )
        )

    selected = [item[3] for item in sorted(selected)]
    if limit <= 0:
        return []
    return selected[:limit]


def analyze(paths, archetypes=None, limit=20, registry_path=REGISTRY):
    registry = load_registry(registry_path)
    declared = normalize_archetypes(archetypes or [], registry)
    scan = scan_sources(paths)
    signal_tags = sorted(scan["signals"])
    suggested = suggest_archetypes(signal_tags)
    patterns = select_patterns(registry, declared, suggested, signal_tags, limit)

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
        "kind": "ascend_operator_design",
        "status": "advisory",
        "declared_archetypes": declared,
        "suggested_archetypes": suggested,
        "archetype_rule": "declared archetypes are hints; static suggestions are not contract facts and may be overridden after reading the task statement",
        "contract_evidence": {
            "task_contract_files": scan["task_contract_files"],
            "public_interface_evidence": scan["public_interface_evidence"],
            "host_source_count": scan["host_source_count"],
            "kernel_source_count": scan["kernel_source_count"],
        },
        "static_signals": scan["signals"],
        "files_scanned": scan["files_scanned"],
        "unknowns": unknowns,
        "decisions": patterns,
        "rule": "Use this checklist to expose missing design decisions. Codex owns the concrete algorithm, task mapping, tiling, buffer sizes and specialization choices; authoritative contract/build/correctness evidence overrides all heuristics.",
    }


def print_report(result):
    print("kind=%s status=%s" % (result["kind"], result["status"]))
    print("declared_archetypes=%s" % (",".join(result["declared_archetypes"]) or "none"))
    print("suggested_archetypes=%s" % (",".join(result["suggested_archetypes"]) or "none"))
    print("files_scanned=%d" % result["files_scanned"])
    print("static_signal_tags=%s" % (",".join(sorted(result["static_signals"])) or "none"))
    print("unknowns=%s" % ("; ".join(result["unknowns"]) or "none"))
    print()
    for index, item in enumerate(result["decisions"], 1):
        print("[%d] %s — %s" % (index, item["id"], item["title"]))
        print("  decide: %s" % item["decide"])
        print("  guardrails: %s" % item["guardrails"])
        print("  validate: %s" % item["validate"])
        if item.get("reasons"):
            print("  selected_by: %s" % "; ".join(item["reasons"]))
        print()
    print("rule=%s" % result["rule"])


def main():
    parser = argparse.ArgumentParser(description="Ascend operator-development design analyzer")
    parser.add_argument("--task")
    parser.add_argument("--task-dir")
    parser.add_argument("--workspace")
    parser.add_argument("--source-dir", action="append", default=[])
    parser.add_argument("--archetype", action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
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
    result = analyze(resolved, archetypes=args.archetype, limit=max(0, args.limit))
    result["source_dirs"] = [display_path(p) for p in resolved]

    if args.as_json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

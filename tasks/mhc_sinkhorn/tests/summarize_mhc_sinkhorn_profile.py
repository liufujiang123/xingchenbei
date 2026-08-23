#!/usr/bin/env python3
"""Summarize only metrics actually exported by MhcSinkhorn msprof runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def numeric(rows: list[dict], column: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(column, "").strip()
        if value:
            values.append(float(value))
    return values


def summary(values: list[float]) -> str:
    return (
        f"median={statistics.median(values):.6f}, mean={statistics.mean(values):.6f}, "
        f"min={min(values):.6f}, max={max(values):.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    summaries = sorted(args.profile_root.rglob("op_summary_*.csv"))
    if not summaries:
        parser.error("op_summary CSV was not exported")
    rows = []
    for path in summaries:
        with path.open(newline="", encoding="utf-8-sig") as source:
            rows.extend(
                row
                for row in csv.DictReader(source)
                if row.get("OP Type") == "MhcSinkhorn"
            )
    if not rows:
        parser.error("no MhcSinkhorn rows in op_summary")

    columns = [
        ("Task Duration(us)", "Task duration (us)", False),
        ("Task Wait Time(us)", "Task wait time (us)", False),
        ("aiv_time(us)", "AIV time (us)", False),
        ("aiv_vec_time(us)", "Vector pipe time (us)", False),
        ("aiv_vec_ratio", "Vector pipe ratio", True),
        ("aiv_scalar_time(us)", "Scalar pipe time (us)", False),
        ("aiv_scalar_ratio", "Scalar pipe ratio", True),
        ("aiv_mte2_time(us)", "MTE2 time (us)", False),
        ("aiv_mte2_ratio", "MTE2 ratio", True),
        ("aiv_mte3_time(us)", "MTE3 time (us)", False),
        ("aiv_mte3_ratio", "MTE3 ratio", True),
        ("aiv_icache_miss_rate", "AIV I-cache miss rate", True),
    ]
    lines = [
        f"# MhcSinkhorn local A3 msprof — {args.label}",
        "",
        "> Evidence class: `profile_observed`.",
        "> CANN 8.5 msprof task-based `PipeUtilization` on local Ascend910_9382; not CANNJudge 910B evidence.",
        "",
        f"- Profiled MhcSinkhorn tasks: {len(rows)}",
        f"- Task type: {rows[0].get('Task Type', 'not exported')}",
        f"- Exported block dimension: {rows[0].get('Block Dim', 'not exported')}",
        "",
        "| Exported metric | Aggregate over profiled tasks |",
        "|---|---|",
    ]
    emitted = []
    for column, label, is_ratio in columns:
        values = numeric(rows, column)
        if not values:
            continue
        rendered = summary(values)
        if is_ratio:
            rendered += f" (median {statistics.median(values) * 100.0:.3f}%)"
        lines.append(f"| {label} | {rendered} |")
        emitted.append((column, values))

    block_dims = sorted(set(numeric(rows, "Block Dim")))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"- Observed block dimensions: {', '.join(f'{value:g}' for value in block_dims)}.",
            "- msprof did not export a direct per-core occupancy or active-core utilization percentage; block dimension is not relabeled as utilization.",
            "- `Task Wait Time` is reported as exported. It is not interpreted as a Vector-pipe stall counter.",
            "- No unexported stall, bandwidth, or utilization metric is inferred here.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    for column, values in emitted:
        marker = column.upper().replace(" ", "_").replace("(", "_").replace(")", "")
        print(f"PROFILE_METRIC_{marker}_MEDIAN={statistics.median(values):.6f}")
    print(f"PROFILE_TASK_COUNT={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

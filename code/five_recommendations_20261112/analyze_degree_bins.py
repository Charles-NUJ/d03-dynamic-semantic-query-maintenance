#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=80)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    degree = Counter(int(x["pageid"]) for x in reference)
    changed = [i for i, row in enumerate(rows) if int(row["input_changed"]) == 1]
    ours = set(sorted(changed, key=lambda i: (-degree[int(rows[i]["pageid"])], i))[: args.budget])
    by_id = set(sorted(changed)[: args.budget])

    bins = [
        ("3--21", 3, 21),
        ("23--32", 23, 32),
        ("35--64", 35, 64),
        ("67--1350", 67, 1350),
    ]
    records = []
    for label, lo, hi in bins:
        cells = [i for i in changed if lo <= degree[int(rows[i]["pageid"])] <= hi]
        transitions = [i for i in cells if int(rows[i]["label_flip"]) == 1]
        for method, selected in (("input_change_id", by_id), ("signature_degree_ours", ours)):
            selected_cells = [i for i in cells if i in selected]
            repaired = sum(degree[int(rows[i]["pageid"])] for i in transitions if i in selected)
            remaining = sum(degree[int(rows[i]["pageid"])] for i in transitions if i not in selected)
            records.append(
                {
                    "degree_bin": label,
                    "method": method,
                    "changed_cells": len(cells),
                    "transition_cells": len(transitions),
                    "transition_fraction": len(transitions) / len(cells),
                    "selected_cells": len(selected_cells),
                    "selected_transition_cells": sum(i in selected for i in transitions),
                    "repaired_transition_rows": repaired,
                    "remaining_transition_rows": remaining,
                }
            )

    csv_path = args.output / "degree_bin_attribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {
        "status": "VERIFIED_POSTHOC",
        "selection_contract": "Both methods use the same 80-call cap and changed set; label_flip is used only after selection.",
        "inputs": {"rows_sha256": sha256(args.rows), "reference_sha256": sha256(args.reference)},
        "changed_cells": len(changed),
        "budget": args.budget,
        "bins": records,
        "checks": {
            "partition_changed_cells_per_method": all(
                sum(r["changed_cells"] for r in records if r["method"] == m) == len(changed)
                for m in ("input_change_id", "signature_degree_ours")
            ),
            "same_transition_population": all(
                records[2 * j]["transition_cells"] == records[2 * j + 1]["transition_cells"]
                for j in range(len(bins))
            ),
            "same_budget": len(ours) == args.budget and len(by_id) == args.budget,
        },
    }
    summary_path = args.output / "degree_bin_attribution.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate the frozen 64-cell gain map by one structural factor at a time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


FACTORS = (
    "target_degree_cv",
    "budget_fraction_of_changed",
    "changed_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 64:
        raise ValueError(f"Expected 64 factor cells, found {len(rows)}")

    records: list[dict[str, object]] = []
    for factor in FACTORS:
        groups: dict[float, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[float(row[factor])].append(row)
        for value in sorted(groups):
            group = groups[value]
            realized = [float(r["realized_gain_percentage_points"]) for r in group]
            robust = [float(r["robust_gain_percentage_points"]) for r in group]
            records.append(
                {
                    "factor": factor,
                    "value": value,
                    "cells": len(group),
                    "mean_realized_gain_pp": statistics.fmean(realized),
                    "median_realized_gain_pp": statistics.median(realized),
                    "min_realized_gain_pp": min(realized),
                    "max_realized_gain_pp": max(realized),
                    "mean_robust_gain_pp": statistics.fmean(robust),
                    "median_robust_gain_pp": statistics.median(robust),
                    "negative_realized_cells": sum(x < -1e-12 for x in realized),
                    "realized_gain_at_most_one_pp_cells": sum(x <= 1.0 + 1e-12 for x in realized),
                }
            )

    csv_path = args.output_dir / "gain_mechanism_by_factor.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    degree_rows = [r for r in records if r["factor"] == "target_degree_cv"]
    budget_rows = [r for r in records if r["factor"] == "budget_fraction_of_changed"]
    summary = {
        "input_csv": str(args.input_csv),
        "input_sha256": sha256(args.input_csv),
        "input_cells": len(rows),
        "interpretation": {
            "degree_heterogeneity": "Median robust and realized gains rise from the uniform-degree boundary as degree CV increases.",
            "budget_position": "A partial budget exposes the ranking choice; full changed-input coverage removes it.",
            "realized_reversals": "All negative realized cells occur at uniform degree, where causal methods share the same robust value and tie-breaking determines realized selection.",
        },
        "degree_cv": degree_rows,
        "budget_fraction": budget_rows,
    }
    json_path = args.output_dir / "gain_mechanism_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        csv_path.name: sha256(csv_path),
        json_path.name: sha256(json_path),
    }
    manifest_path = args.output_dir / "sha256_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

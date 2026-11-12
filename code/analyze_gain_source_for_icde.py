#!/usr/bin/env python3
"""Derive the ICDE gain-source audit from the frozen structure map.

This script performs no semantic inference.  It compares the proposed method
with three causal single-mechanism controls under identical factor settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


OURS = "signature_degree_ours"
CONTROLS = ("input_change_id", "degree_all", "random_changed")
KEYS = ("changed_fraction", "target_degree_cv", "budget_fraction_of_changed")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open(newline="", encoding="utf-8-sig") as src:
        rows = list(csv.DictReader(src))

    grouped: dict[tuple[float, float, float], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = tuple(f(row, k) for k in KEYS)
        grouped.setdefault(key, {})[row["method"]] = row

    derived: list[dict[str, object]] = []
    for key in sorted(grouped):
        methods = grouped[key]
        missing = [m for m in (OURS,) + CONTROLS if m not in methods]
        if missing:
            raise ValueError(f"Missing methods {missing} for {key}")
        ours = methods[OURS]
        best_robust = min(CONTROLS, key=lambda m: f(methods[m], "mean_minimax_residual_fraction"))
        best_realized = min(CONTROLS, key=lambda m: f(methods[m], "mean_realized_join_error_fraction"))
        record = {
            "changed_fraction": key[0],
            "target_degree_cv": key[1],
            "budget_fraction_of_changed": key[2],
            "repetitions": int(float(ours["repetitions"])),
            "ours_robust_fraction": f(ours, "mean_minimax_residual_fraction"),
            "best_control_robust_method": best_robust,
            "best_control_robust_fraction": f(methods[best_robust], "mean_minimax_residual_fraction"),
            "robust_gain_percentage_points": 100.0 * (
                f(methods[best_robust], "mean_minimax_residual_fraction")
                - f(ours, "mean_minimax_residual_fraction")
            ),
            "ours_realized_fraction": f(ours, "mean_realized_join_error_fraction"),
            "best_control_realized_method": best_realized,
            "best_control_realized_fraction": f(methods[best_realized], "mean_realized_join_error_fraction"),
            "realized_gain_percentage_points": 100.0 * (
                f(methods[best_realized], "mean_realized_join_error_fraction")
                - f(ours, "mean_realized_join_error_fraction")
            ),
        }
        derived.append(record)

    def pick(cf: float, cv: float, bf: float) -> dict[str, object]:
        return next(
            r
            for r in derived
            if r["changed_fraction"] == cf
            and r["target_degree_cv"] == cv
            and r["budget_fraction_of_changed"] == bf
        )

    selected = {
        "amplified_sparse_heterogeneous_partial_budget": pick(0.1, 1.88, 0.25),
        "trace_matched_structure": pick(0.453125, 1.88, 0.55),
        "uniform_degree_boundary": pick(1.0, 0.0, 0.1),
        "full_changed_budget_boundary": pick(0.453125, 1.88, 1.0),
    }
    summary = {
        "source_csv": str(args.input_csv),
        "source_sha256": sha256(args.input_csv),
        "source_rows": len(rows),
        "factor_cells": len(derived),
        "controls": list(CONTROLS),
        "comparison_contract": "same changed fraction, degree CV, changed-budget fraction, and 200 assignments",
        "cells_with_negative_realized_gain": sum(
            float(r["realized_gain_percentage_points"]) < -1e-12 for r in derived
        ),
        "cells_with_realized_gain_at_most_one_point": sum(
            float(r["realized_gain_percentage_points"]) <= 1.0 + 1e-12 for r in derived
        ),
        "selected_regimes": selected,
    }

    detail_path = args.output_dir / "gain_source_all_cells.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(derived[0]))
        writer.writeheader()
        writer.writerows(derived)

    summary_path = args.output_dir / "gain_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "gain_source_all_cells.csv": sha256(detail_path),
        "gain_source_summary.json": sha256(summary_path),
    }
    (args.output_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

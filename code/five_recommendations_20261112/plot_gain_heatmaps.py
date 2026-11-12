#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_prefix", type=Path)
    args = p.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    changes = sorted({float(r["changed_fraction"]) for r in rows})
    cvs = sorted({float(r["target_degree_cv"]) for r in rows})
    budgets = sorted({float(r["budget_fraction_of_changed"]) for r in rows})
    lookup = {(float(r["changed_fraction"]), float(r["target_degree_cv"]), float(r["budget_fraction_of_changed"])): r for r in rows}

    plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.35), constrained_layout=True)
    metrics = (("robust_gain_percentage_points", "Worst-case gain (pp)"), ("realized_gain_percentage_points", "Realized gain (pp)"))
    for row_idx, (metric, row_title) in enumerate(metrics):
        values = [float(r[metric]) for r in rows]
        vmin, vmax = min(values), max(values)
        for col_idx, change in enumerate(changes):
            ax = axes[row_idx, col_idx]
            matrix = np.array([[float(lookup[(change, cv, budget)][metric]) for budget in budgets] for cv in cvs])
            ax.imshow(matrix, cmap="Greys", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            for y in range(len(cvs)):
                for x in range(len(budgets)):
                    val = matrix[y, x]
                    color = "white" if val > vmin + 0.58 * (vmax - vmin) else "black"
                    label = f"{val:.1f}" + ("*" if val < -1e-10 else "")
                    ax.text(x, y, label, ha="center", va="center", fontsize=6.4, color=color)
            ax.set_xticks(range(len(budgets)), [f"{x:g}" for x in budgets])
            ax.set_yticks(range(len(cvs)), [f"{x:g}" for x in cvs] if col_idx == 0 else [])
            ax.set_title(f"Changed fraction {change:.2f}", fontsize=7.4)
            if row_idx == 1:
                ax.set_xlabel("Budget / changed inputs")
            if col_idx == 0:
                ax.set_ylabel(row_title + "\nDegree CV")
    fig.text(0.5, -0.005, "Each value compares our rule with the best single-mechanism causal control; * marks a realized reversal.", ha="center", fontsize=7)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

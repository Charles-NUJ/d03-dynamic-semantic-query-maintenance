"""Recompute the D03 heterogeneous-token table from frozen local inputs.

This is a content audit.  It performs no model calls and creates no figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from solve_signature_knapsack import select_signature_knapsack


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def greedy(order: list[int], costs: list[int], budget: int) -> tuple[int, ...]:
    selected: list[int] = []
    used = 0
    for index in order:
        if used + costs[index] <= budget:
            selected.append(index)
            used += costs[index]
    return tuple(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--random-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = json.loads(args.reference.read_text(encoding="utf-8"))
    prediction_by_key = {
        (int(item["revid"]), int(item["predicate_index"])): item
        for item in predictions
    }
    links: dict[int, set[str]] = defaultdict(set)
    for item in references:
        links[int(item["pageid"])].add(str(item["target"]))

    cells: list[dict[str, int]] = []
    for row in rows:
        degree = len(links[int(row["pageid"])])
        if int(row["input_changed"]) and degree > 0:
            prediction = prediction_by_key[
                (int(row["new_revid"]), int(row["predicate_index"]))
            ]
            cells.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "degree": degree,
                    "cost": min(
                        int(prediction["input_tokens_untruncated"]), args.max_tokens
                    ),
                    "transition": int(row["label_flip"]),
                }
            )

    values = [cell["degree"] for cell in cells]
    costs = [cell["cost"] for cell in cells]
    transitions = [cell["transition"] for cell in cells]
    total_degree = sum(values)
    full_join_mass = sum(
        len(links[int(row["pageid"])]) * int(row["new_label"])
        for row in rows
    )
    degree_order = sorted(range(len(cells)), key=lambda i: (-values[i], i))
    density_order = sorted(
        range(len(cells)), key=lambda i: (-(values[i] / costs[i]), -values[i], i)
    )
    id_order = list(range(len(cells)))
    prefixes = (32, 80, 112)
    budgets = [sum(costs[i] for i in degree_order[:count]) for count in prefixes]

    def score(selected: tuple[int, ...]) -> dict[str, float | int]:
        chosen = set(selected)
        residual_degree = sum(
            value for index, value in enumerate(values) if index not in chosen
        )
        join_error = sum(
            value * transitions[index]
            for index, value in enumerate(values)
            if index not in chosen
        )
        return {
            "calls": len(selected),
            "used_cost": sum(costs[index] for index in selected),
            "robust_residual_fraction": residual_degree / total_degree,
            "join_error_fraction": join_error / full_join_mass,
        }

    records: list[dict[str, float | int | str]] = []
    dp_mismatches = 0
    for budget_index, budget in enumerate(budgets):
        started = time.perf_counter()
        exact = select_signature_knapsack(
            [True] * len(cells), values, costs, int(budget)
        )
        dp_seconds = time.perf_counter() - started
        deterministic = {
            "signature_knapsack_dp_ours": tuple(exact.selected_indices),
            "degree_per_cost_ours": greedy(density_order, costs, budget),
            "largest_degree_ours": greedy(degree_order, costs, budget),
            "identifier_control_ours": greedy(id_order, costs, budget),
        }
        exact_removed = exact.removed_worst_case_degree
        for method, selected in deterministic.items():
            item = score(selected)
            removed = total_degree - round(
                float(item["robust_residual_fraction"]) * total_degree
            )
            if method == "signature_knapsack_dp_ours" and removed != exact_removed:
                dp_mismatches += 1
            records.append(
                {
                    "budget": budget,
                    "method": method,
                    "repetitions": 1,
                    "selection_seconds": dp_seconds
                    if method == "signature_knapsack_dp_ours"
                    else 0.0,
                    **item,
                }
            )

        random_scores = []
        for repetition in range(args.random_repetitions):
            order = list(range(len(cells)))
            random.Random(
                args.seed + 10000 * prefixes[budget_index] + repetition
            ).shuffle(order)
            random_scores.append(score(greedy(order, costs, budget)))
        records.append(
            {
                "budget": budget,
                "method": "random_control_ours",
                "repetitions": args.random_repetitions,
                "selection_seconds": 0.0,
                "calls": statistics.fmean(float(x["calls"]) for x in random_scores),
                "used_cost": statistics.fmean(
                    float(x["used_cost"]) for x in random_scores
                ),
                "robust_residual_fraction": statistics.fmean(
                    float(x["robust_residual_fraction"]) for x in random_scores
                ),
                "join_error_fraction": statistics.fmean(
                    float(x["join_error_fraction"]) for x in random_scores
                ),
            }
        )

    payload = {
        "status": "VERIFIED" if dp_mismatches == 0 else "FAILED",
        "scope": {
            "changed_positive_degree_cells": len(cells),
            "total_changed_degree": total_degree,
            "full_join_mass": full_join_mass,
            "max_tokens": args.max_tokens,
            "random_repetitions": args.random_repetitions,
            "seed": args.seed,
            "degree_prefix_counts": list(prefixes),
            "budgets": budgets,
        },
        "records": records,
        "checks": {"dp_internal_value_mismatches": dp_mismatches},
        "inputs": {
            str(args.rows): sha256(args.rows),
            str(args.predictions): sha256(args.predictions),
            str(args.reference): sha256(args.reference),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

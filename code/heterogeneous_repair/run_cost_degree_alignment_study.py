"""Controlled cost--degree alignment study for D03 heterogeneous repair.

The study holds the changed cells, join degrees, cost multiset, and budget
fractions fixed.  It changes only which observed cost is paired with each join
degree.  This isolates the structural source of any gain from exact knapsack
allocation over one-pass orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import spearmanr

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


def score(
    selected: tuple[int, ...], values: list[int], costs: list[int], budget: int
) -> dict[str, float | int]:
    removed = sum(values[index] for index in selected)
    total = sum(values)
    used = sum(costs[index] for index in selected)
    if used > budget:
        raise AssertionError((used, budget))
    return {
        "selected_count": len(selected),
        "used_cost": used,
        "removed_degree": removed,
        "robust_residual_fraction": (total - removed) / total,
    }


def milp_optimum(values: list[int], costs: list[int], budget: int) -> int:
    result = milp(
        c=-np.asarray(values, dtype=float),
        integrality=np.ones(len(values), dtype=int),
        bounds=Bounds(np.zeros(len(values)), np.ones(len(values))),
        constraints=LinearConstraint(
            np.asarray([costs], dtype=float),
            np.asarray([-np.inf]),
            np.asarray([float(budget)]),
        ),
        options={"mip_rel_gap": 0.0, "time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return int(round(-float(result.fun)))


def load_natural_cells(
    rows_path: Path,
    predictions_path: Path,
    reference_path: Path,
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prediction_by_key = {
        (int(item["revid"]), int(item["predicate_index"])): item
        for item in predictions
    }
    links: dict[int, set[str]] = defaultdict(set)
    for item in reference:
        links[int(item["pageid"])].add(str(item["target"]))

    values: list[int] = []
    costs: list[int] = []
    for row in rows:
        degree = len(links[int(row["pageid"])])
        if int(row["input_changed"]) and degree > 0:
            prediction = prediction_by_key[
                (int(row["new_revid"]), int(row["predicate_index"]))
            ]
            token_cost = min(int(prediction["input_tokens_untruncated"]), max_tokens)
            if token_cost <= 0:
                raise ValueError(token_cost)
            values.append(degree)
            costs.append(token_cost)
    return values, costs


def make_pairings(values: list[int], observed: list[int]) -> dict[str, list[int]]:
    degree_order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ascending_costs = sorted(observed)
    descending_costs = list(reversed(ascending_costs))

    positive = [0] * len(values)
    negative = [0] * len(values)
    for rank, index in enumerate(degree_order):
        positive[index] = descending_costs[rank]
        negative[index] = ascending_costs[rank]

    shuffled = list(observed)
    random.Random(20261112).shuffle(shuffled)
    return {
        "observed": list(observed),
        "high_degree_high_cost": positive,
        "high_degree_low_cost": negative,
        "fixed_seed_shuffle": shuffled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--random-repetitions", type=int, default=1000)
    args = parser.parse_args()

    values, observed_costs = load_natural_cells(
        args.rows, args.predictions, args.reference, args.max_tokens
    )
    pairings = make_pairings(values, observed_costs)
    sorted_multiset = sorted(observed_costs)
    if any(sorted(costs) != sorted_multiset for costs in pairings.values()):
        raise AssertionError("cost multiset changed")

    budget_fractions = (0.2, 0.4, 0.6, 0.8)
    total_cost = sum(observed_costs)
    degree_order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    id_order = list(range(len(values)))
    records: list[dict[str, float | int | str]] = []
    dp_milp_mismatches = 0

    for pairing_name, costs in pairings.items():
        density_order = sorted(
            range(len(values)),
            key=lambda index: (-(values[index] / costs[index]), -values[index], index),
        )
        for fraction in budget_fractions:
            budget = int(math.floor(total_cost * fraction))
            started = time.perf_counter()
            exact = select_signature_knapsack(
                [True] * len(values), values, costs, budget
            )
            dp_seconds = time.perf_counter() - started
            milp_value = milp_optimum(values, costs, budget)
            if exact.removed_worst_case_degree != milp_value:
                dp_milp_mismatches += 1

            deterministic = {
                "signature_knapsack_dp_ours": tuple(exact.selected_indices),
                "largest_degree_ours": greedy(degree_order, costs, budget),
                "degree_per_cost_ours": greedy(density_order, costs, budget),
                "identifier_control_ours": greedy(id_order, costs, budget),
            }
            exact_value = exact.removed_worst_case_degree
            for method, selected in deterministic.items():
                result = score(selected, values, costs, budget)
                records.append(
                    {
                        "pairing": pairing_name,
                        "budget_fraction": fraction,
                        "budget": budget,
                        "method": method,
                        "repetitions": 1,
                        "selection_seconds": dp_seconds if method == "signature_knapsack_dp_ours" else 0.0,
                        "optimality_gap_fraction":
                            (exact_value - int(result["removed_degree"])) / exact_value,
                        **result,
                    }
                )

            random_scores = []
            for repetition in range(args.random_repetitions):
                order = list(range(len(values)))
                random.Random(
                    20261112
                    + 100_000 * list(pairings).index(pairing_name)
                    + 1_000 * int(round(fraction * 10))
                    + repetition
                ).shuffle(order)
                random_scores.append(score(greedy(order, costs, budget), values, costs, budget))
            mean_removed = statistics.fmean(
                float(item["removed_degree"]) for item in random_scores
            )
            records.append(
                {
                    "pairing": pairing_name,
                    "budget_fraction": fraction,
                    "budget": budget,
                    "method": "random_control_ours",
                    "repetitions": args.random_repetitions,
                    "selection_seconds": 0.0,
                    "selected_count": statistics.fmean(
                        float(item["selected_count"]) for item in random_scores
                    ),
                    "used_cost": statistics.fmean(
                        float(item["used_cost"]) for item in random_scores
                    ),
                    "removed_degree": mean_removed,
                    "robust_residual_fraction": statistics.fmean(
                        float(item["robust_residual_fraction"])
                        for item in random_scores
                    ),
                    "optimality_gap_fraction": (exact_value - mean_removed) / exact_value,
                }
            )

    summaries = []
    for pairing_name, costs in pairings.items():
        subset = [row for row in records if row["pairing"] == pairing_name]
        correlation = float(spearmanr(values, costs).statistic)
        summary: dict[str, float | int | str] = {
            "pairing": pairing_name,
            "spearman_degree_cost": correlation,
            "cost_min": min(costs),
            "cost_median": statistics.median(costs),
            "cost_max": max(costs),
            "cost_cv": statistics.pstdev(costs) / statistics.fmean(costs),
        }
        for method in (
            "largest_degree_ours",
            "degree_per_cost_ours",
            "identifier_control_ours",
            "random_control_ours",
        ):
            method_rows = [row for row in subset if row["method"] == method]
            summary[f"{method}_mean_gap"] = statistics.fmean(
                float(row["optimality_gap_fraction"]) for row in method_rows
            )
            summary[f"{method}_max_gap"] = max(
                float(row["optimality_gap_fraction"]) for row in method_rows
            )
        summaries.append(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "status": "VERIFIED" if dp_milp_mismatches == 0 else "FAILED",
        "scientific_question": (
            "How does the pairing between observable call cost and current join "
            "degree determine the value of exact heterogeneous allocation?"
        ),
        "controlled_contract": {
            "fixed": [
                "145 changed positive-degree cells",
                "join-degree vector",
                "observed capped-token cost multiset",
                "budget fractions",
                "robust residual definition",
            ],
            "varied": "the permutation that pairs costs with join degrees",
        },
        "candidate_cells": len(values),
        "total_degree": sum(values),
        "total_cost": total_cost,
        "budget_fractions": list(budget_fractions),
        "random_repetitions": args.random_repetitions,
        "dp_milp_checks": len(pairings) * len(budget_fractions),
        "dp_milp_mismatches": dp_milp_mismatches,
        "pairing_summaries": summaries,
        "records": records,
        "inputs": {
            str(path): sha256(path)
            for path in (args.rows, args.predictions, args.reference)
        },
        "script_sha256": sha256(Path(__file__)),
        "boundary": (
            "The cost permutations are controlled mechanisms, not additional "
            "production traces. They explain when exact allocation matters under "
            "the same positive additive-cost model."
        ),
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "candidate_cells": len(values),
        "dp_milp_mismatches": dp_milp_mismatches,
        "pairing_summaries": summaries,
    }, indent=2))
    if payload["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

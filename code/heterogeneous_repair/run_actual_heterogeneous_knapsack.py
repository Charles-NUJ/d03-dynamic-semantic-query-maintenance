"""Exact heterogeneous-cost repair on the current 320-cell D03 event."""

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

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_dp(values: list[int], costs: list[int], budget: int):
    """Pseudo-polynomial exact 0-1 knapsack for positive integer costs."""
    n = len(values)
    dp = np.zeros(budget + 1, dtype=np.int64)
    take = np.zeros((n, budget + 1), dtype=np.bool_)
    started = time.perf_counter()
    for i, (value, cost) in enumerate(zip(values, costs)):
        if cost <= 0:
            raise ValueError("every cost must be positive")
        previous = dp.copy()
        candidate = previous[: budget + 1 - cost] + value
        target = dp[cost:]
        improved = candidate > target
        target[improved] = candidate[improved]
        take[i, cost:][improved] = True
    selected = []
    remaining = budget
    for i in range(n - 1, -1, -1):
        if take[i, remaining]:
            selected.append(i)
            remaining -= costs[i]
    selected.reverse()
    return int(dp[budget]), tuple(selected), time.perf_counter() - started


def exact_milp(values: list[int], costs: list[int], budget: int):
    started = time.perf_counter()
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
    selected = tuple(i for i, value in enumerate(result.x) if value >= 0.5)
    return sum(values[i] for i in selected), selected, time.perf_counter() - started, result


def greedy(order, costs: list[int], budget: int):
    selected = []
    used = 0
    for i in order:
        if used + costs[i] <= budget:
            selected.append(i)
            used += costs[i]
    return tuple(selected)


def metrics(selected, values, costs, flips, full_join_mass):
    chosen = set(selected)
    removed = sum(values[i] for i in chosen)
    total = sum(values)
    actual_error_rows = sum(values[i] * flips[i] for i in range(len(values)) if i not in chosen)
    return {
        "selected_count": len(chosen),
        "used_tokens": sum(costs[i] for i in chosen),
        "removed_worst_case_degree": removed,
        "worst_case_residual_fraction": (total - removed) / total,
        "actual_join_error_rows": actual_error_rows,
        "actual_join_error_fraction": actual_error_rows / full_join_mass,
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

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    prediction_by_key = {
        (int(item["revid"]), int(item["predicate_index"])): item
        for item in predictions
    }
    links = defaultdict(set)
    for item in reference:
        links[int(item["pageid"])].add(item["target"])

    candidates = []
    full_join_mass = 0
    for row_index, row in enumerate(rows):
        degree = len(links[int(row["pageid"])])
        full_join_mass += int(row["new_label"]) * degree
        if int(row["input_changed"]) and degree > 0:
            prediction = prediction_by_key[(int(row["new_revid"]), int(row["predicate_index"]))]
            token_cost = min(int(prediction["input_tokens_untruncated"]), args.max_tokens)
            if token_cost <= 0:
                raise ValueError((row_index, token_cost))
            candidates.append({
                "row_index": row_index,
                "degree": degree,
                "token_cost": token_cost,
                "flip": int(row["label_flip"]),
            })

    values = [item["degree"] for item in candidates]
    costs = [item["token_cost"] for item in candidates]
    flips = [item["flip"] for item in candidates]
    degree_order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    density_order = sorted(
        range(len(values)),
        key=lambda i: (-(values[i] / costs[i]), -values[i], i),
    )
    id_order = list(range(len(values)))

    budget_specs = []
    for call_equivalent in (32, 80, 112):
        selected = degree_order[:call_equivalent]
        budget_specs.append({
            "name": f"tokens_of_top_{call_equivalent}_unit_calls",
            "call_equivalent": call_equivalent,
            "token_budget": sum(costs[i] for i in selected),
        })

    records = []
    budget_summaries = []
    for budget_spec in budget_specs:
        budget = budget_spec["token_budget"]
        dp_value, dp_selected, dp_seconds = exact_dp(values, costs, budget)
        milp_value, milp_selected, milp_seconds, solved = exact_milp(values, costs, budget)
        if dp_value != milp_value:
            raise AssertionError((budget_spec, dp_value, milp_value))

        deterministic = {
            "signature_knapsack_dp_ours": (dp_selected, dp_seconds),
            "signature_knapsack_milp_check": (milp_selected, milp_seconds),
            "top_degree_under_token_budget": (greedy(degree_order, costs, budget), 0.0),
            "degree_per_token_greedy": (greedy(density_order, costs, budget), 0.0),
            "input_change_id_under_token_budget": (greedy(id_order, costs, budget), 0.0),
            "unit_cost_top_k": (tuple(degree_order[: budget_spec["call_equivalent"]]), 0.0),
        }
        for method, (selected, seconds) in deterministic.items():
            result = {
                "budget_name": budget_spec["name"],
                "call_equivalent": budget_spec["call_equivalent"],
                "token_budget": budget,
                "method": method,
                "selection_seconds": seconds,
                "repetitions": 1,
                **metrics(selected, values, costs, flips, full_join_mass),
            }
            result["robust_optimality_gap_fraction"] = (dp_value - result["removed_worst_case_degree"]) / dp_value
            records.append(result)

        random_metrics = []
        for repeat in range(args.random_repetitions):
            order = list(range(len(values)))
            random.Random(20261112 + 10000 * budget_spec["call_equivalent"] + repeat).shuffle(order)
            random_metrics.append(metrics(greedy(order, costs, budget), values, costs, flips, full_join_mass))
        random_record = {
            "budget_name": budget_spec["name"],
            "call_equivalent": budget_spec["call_equivalent"],
            "token_budget": budget,
            "method": "random_token_budget",
            "selection_seconds": 0.0,
            "repetitions": args.random_repetitions,
        }
        for field in random_metrics[0]:
            random_record[field] = statistics.fmean(item[field] for item in random_metrics)
        random_record["robust_optimality_gap_fraction"] = (
            dp_value - random_record["removed_worst_case_degree"]
        ) / dp_value
        records.append(random_record)

        budget_summaries.append({
            **budget_spec,
            "dp_value": dp_value,
            "milp_value": milp_value,
            "dp_milp_match": dp_value == milp_value,
            "dp_seconds": dp_seconds,
            "milp_seconds": milp_seconds,
            "milp_status": int(solved.status),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in records for field in row})
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "status": "VERIFIED" if all(item["dp_milp_match"] for item in budget_summaries) else "FAILED",
        "model": "positive-integer token-cost 0-1 knapsack over positive-degree changed inputs",
        "affected_cells": len(rows),
        "candidate_cells": len(candidates),
        "total_candidate_tokens": sum(costs),
        "token_cost": {
            "minimum": min(costs),
            "median": statistics.median(costs),
            "maximum": max(costs),
            "mean": statistics.fmean(costs),
            "coefficient_of_variation": statistics.pstdev(costs) / statistics.fmean(costs),
            "strictly_positive": all(cost > 0 for cost in costs),
        },
        "full_join_mass": full_join_mass,
        "budgets": budget_summaries,
        "records": records,
        "inputs": {
            str(path): sha256(path)
            for path in (args.rows, args.predictions, args.reference)
        },
        "boundary": "Token count is an observed additive cost proxy. Remote queueing, monetary price, and parallel batch makespan are outside this experiment.",
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "candidate_cells": len(candidates),
        "token_cost": payload["token_cost"],
        "budgets": budget_summaries,
    }, indent=2))
    if payload["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Exhaustively verify the small positive-cost D03 knapsack model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brute_force(values: tuple[int, ...], costs: tuple[int, ...], budget: int):
    best_value = -1
    best_mask = 0
    for mask in range(1 << len(values)):
        used = sum(costs[i] for i in range(len(values)) if mask & (1 << i))
        if used > budget:
            continue
        value = sum(values[i] for i in range(len(values)) if mask & (1 << i))
        if value > best_value or (value == best_value and mask < best_mask):
            best_value = value
            best_mask = mask
    return best_value, best_mask


def dynamic_program(values: tuple[int, ...], costs: tuple[int, ...], budget: int):
    previous = [0] * (budget + 1)
    for value, cost in zip(values, costs):
        current = previous.copy()
        for b in range(cost, budget + 1):
            current[b] = max(previous[b], previous[b - cost] + value)
        previous = current
    return previous[budget]


def greedy(order: list[int], values: tuple[int, ...], costs: tuple[int, ...], budget: int):
    used = 0
    value = 0
    selected = []
    for i in order:
        if used + costs[i] <= budget:
            used += costs[i]
            value += values[i]
            selected.append(i)
    return value, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checked = 0
    mismatches = []
    heuristic = {
        "largest_degree": {"suboptimal": 0, "sum_gap": 0.0, "worst_gap": -1.0, "case": None},
        "degree_per_cost": {"suboptimal": 0, "sum_gap": 0.0, "worst_gap": -1.0, "case": None},
    }
    for values in itertools.product(range(1, 5), repeat=4):
        for costs in itertools.product(range(1, 4), repeat=4):
            for budget in range(1, sum(costs)):
                checked += 1
                optimum, mask = brute_force(values, costs, budget)
                dp_value = dynamic_program(values, costs, budget)
                if optimum != dp_value:
                    mismatches.append({
                        "values": values,
                        "costs": costs,
                        "budget": budget,
                        "brute_force": optimum,
                        "dynamic_program": dp_value,
                    })
                orders = {
                    "largest_degree": sorted(range(4), key=lambda i: (-values[i], i)),
                    "degree_per_cost": sorted(
                        range(4), key=lambda i: (-(values[i] / costs[i]), -values[i], i)
                    ),
                }
                for name, order in orders.items():
                    got, selected = greedy(order, values, costs, budget)
                    gap = 0.0 if optimum == 0 else (optimum - got) / optimum
                    item = heuristic[name]
                    item["sum_gap"] += gap
                    if gap > 0:
                        item["suboptimal"] += 1
                    if gap > item["worst_gap"]:
                        item["worst_gap"] = gap
                        item["case"] = {
                            "values": list(values),
                            "costs": list(costs),
                            "budget": budget,
                            "optimum": optimum,
                            "optimum_mask": mask,
                            "heuristic_value": got,
                            "heuristic_selected": selected,
                        }

    for item in heuristic.values():
        item["mean_gap"] = item.pop("sum_gap") / checked
        item["suboptimal_fraction"] = item["suboptimal"] / checked

    payload = {
        "status": "VERIFIED" if not mismatches else "FAILED",
        "model": "four positive-value cells; values 1..4; costs 1..3; every binding integer budget",
        "instances_checked": checked,
        "dynamic_program_brute_force_mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
        "heuristics": heuristic,
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

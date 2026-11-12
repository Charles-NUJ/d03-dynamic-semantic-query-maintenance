"""Exact positive-integer-cost selector for D03 input-signature repair."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    removed_worst_case_degree: int
    used_cost: int
    budget: int
    branch: str


def select_signature_knapsack(
    changed: Sequence[bool],
    degrees: Sequence[int],
    costs: Sequence[int],
    budget: int,
) -> SelectionResult:
    """Select changed positive-degree cells under an additive integer budget."""
    if not (len(changed) == len(degrees) == len(costs)):
        raise ValueError("changed, degrees, and costs must have equal length")
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if any(not isinstance(cost, int) or cost <= 0 for cost in costs):
        raise ValueError("every cost must be a positive integer")
    if any(not isinstance(degree, int) or degree < 0 for degree in degrees):
        raise ValueError("every degree must be a nonnegative integer")

    candidates = [
        i for i, (is_changed, degree) in enumerate(zip(changed, degrees))
        if is_changed and degree > 0
    ]
    if not candidates or budget == 0:
        return SelectionResult((), 0, 0, budget, "empty")

    if all(costs[i] == 1 for i in candidates):
        count = min(budget, len(candidates))
        selected = tuple(sorted(candidates, key=lambda i: (-degrees[i], i))[:count])
        return SelectionResult(
            selected,
            sum(degrees[i] for i in selected),
            len(selected),
            budget,
            "unit_cost_degree",
        )

    values = [degrees[i] for i in candidates]
    item_costs = [costs[i] for i in candidates]
    dp = np.zeros(budget + 1, dtype=np.int64)
    take = np.zeros((len(candidates), budget + 1), dtype=np.bool_)
    for item, (value, cost) in enumerate(zip(values, item_costs)):
        if cost > budget:
            continue
        previous = dp.copy()
        candidate_values = previous[: budget + 1 - cost] + value
        improved = candidate_values > dp[cost:]
        dp[cost:][improved] = candidate_values[improved]
        take[item, cost:][improved] = True

    remaining = budget
    selected_local = []
    for item in range(len(candidates) - 1, -1, -1):
        if take[item, remaining]:
            selected_local.append(item)
            remaining -= item_costs[item]
    selected = tuple(sorted(candidates[item] for item in selected_local))
    return SelectionResult(
        selected,
        sum(degrees[i] for i in selected),
        sum(costs[i] for i in selected),
        budget,
        "positive_integer_dp",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = select_signature_knapsack(
        payload["changed"], payload["degrees"], payload["costs"], payload["budget"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

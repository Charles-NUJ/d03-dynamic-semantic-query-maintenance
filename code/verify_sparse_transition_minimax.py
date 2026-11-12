"""Finite audit for cardinality-capped hidden semantic transitions.

This checker tests a stronger form of the D03 minimax claim.  The adversary may
activate at most gamma changed-input cells.  The policy still observes only the
input-change indicators and current join degrees before choosing calls.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsets_with_budget(n: int, budget: int):
    for size in range(budget + 1):
        yield from itertools.combinations(range(n), size)


def allowed_deltas(changed, gamma: int):
    for delta in itertools.product((0, 1), repeat=len(changed)):
        if sum(delta) > gamma:
            continue
        if any(delta[i] > changed[i] for i in range(len(changed))):
            continue
        yield delta


def adversarial_error(degrees, changed, selected, gamma: int):
    selected = set(selected)
    return max(
        sum(
            degrees[i] * delta[i]
            for i in range(len(degrees))
            if i not in selected
        )
        for delta in allowed_deltas(changed, gamma)
    )


def order_statistic_error(degrees, changed, selected, gamma: int):
    selected = set(selected)
    residual = sorted(
        (
            degrees[i]
            for i in range(len(degrees))
            if changed[i] and i not in selected
        ),
        reverse=True,
    )
    return sum(residual[:gamma])


def robust_selection(degrees, changed, budget: int):
    candidates = [
        i for i, value in enumerate(changed)
        if value and degrees[i] > 0
    ]
    candidates.sort(key=lambda i: (-degrees[i], i))
    return tuple(candidates[:budget])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n = 4
    instances = 0
    feasible_actions = 0
    adversary_patterns = 0
    simultaneous_optima = 0
    for degrees in itertools.product(range(4), repeat=n):
        for changed in itertools.product((0, 1), repeat=n):
            for budget in range(n + 1):
                our = robust_selection(degrees, changed, budget)
                our_curve = []
                for gamma in range(n + 1):
                    instances += 1
                    action_errors = []
                    gamma_patterns = sum(1 for _ in allowed_deltas(changed, gamma))
                    for selected in subsets_with_budget(n, budget):
                        feasible_actions += 1
                        adversary_patterns += gamma_patterns
                        enumerated = adversarial_error(
                            degrees, changed, selected, gamma
                        )
                        closed_form = order_statistic_error(
                            degrees, changed, selected, gamma
                        )
                        if enumerated != closed_form:
                            raise AssertionError(
                                (
                                    "order-statistic identity",
                                    degrees,
                                    changed,
                                    budget,
                                    gamma,
                                    selected,
                                    enumerated,
                                    closed_form,
                                )
                            )
                        action_errors.append(enumerated)
                    optimum = min(action_errors)
                    achieved = adversarial_error(
                        degrees, changed, our, gamma
                    )
                    if achieved != optimum:
                        raise AssertionError(
                            (
                                "simultaneous robust optimum",
                                degrees,
                                changed,
                                budget,
                                gamma,
                                our,
                                achieved,
                                optimum,
                            )
                        )
                    our_curve.append(achieved)
                    simultaneous_optima += 1

                if any(a > b for a, b in zip(our_curve, our_curve[1:])):
                    raise AssertionError(
                        ("gamma monotonicity", degrees, changed, budget, our_curve)
                    )
                full_gamma = sum(
                    degrees[i]
                    for i in range(n)
                    if changed[i] and i not in set(our)
                )
                if our_curve[-1] != full_gamma:
                    raise AssertionError(
                        ("uncapped recovery", degrees, changed, budget, our_curve)
                    )

    payload = {
        "status": "VERIFIED",
        "claim": (
            "Selecting the largest positive join degrees among changed-input "
            "cells minimizes worst-case residual join-row error for every "
            "transition-cardinality cap gamma."
        ),
        "scope": {
            "semantic_cells": n,
            "degree_values": [0, 1, 2, 3],
            "input_change_patterns": 2 ** n,
            "budgets": list(range(n + 1)),
            "transition_caps_gamma": list(range(n + 1)),
        },
        "instances": instances,
        "feasible_actions_checked": feasible_actions,
        "adversary_patterns_checked": adversary_patterns,
        "simultaneous_optimality_checks": simultaneous_optima,
        "checks": {
            "cardinality_capped_order_statistic_identity": True,
            "top_degree_optimal_for_every_gamma": True,
            "worst_case_nondecreasing_in_gamma": True,
            "uncapped_theorem_recovered_at_gamma_n": True,
        },
        "scientific_boundary": (
            "Finite exhaustive verification supports but does not replace the "
            "general exchange/majorization proof. Gamma limits the number of "
            "hidden semantic transitions; it is not a control variable."
        ),
        "script_sha256_before_output": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

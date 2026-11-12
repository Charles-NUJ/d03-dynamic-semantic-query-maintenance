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


def adversarial_error(degrees, changed, selected) -> int:
    selected = set(selected)
    worst = 0
    for delta in itertools.product((0, 1), repeat=len(degrees)):
        if any(delta[i] > changed[i] for i in range(len(degrees))):
            continue
        error = sum(
            degrees[i] * delta[i]
            for i in range(len(degrees))
            if i not in selected
        )
        worst = max(worst, error)
    return worst


def formula_error(degrees, changed, selected) -> int:
    selected = set(selected)
    return sum(
        degrees[i] * changed[i]
        for i in range(len(degrees))
        if i not in selected
    )


def our_selection(degrees, changed, budget: int):
    candidates = [i for i, value in enumerate(changed) if value]
    candidates.sort(key=lambda i: (-degrees[i], i))
    return tuple(candidates[:budget])


def degree_selection(degrees, budget: int):
    candidates = list(range(len(degrees)))
    candidates.sort(key=lambda i: (-degrees[i], i))
    return tuple(candidates[:budget])


def changed_id_selection(changed, budget: int):
    return tuple(i for i, value in enumerate(changed) if value)[:budget]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n = 4
    instances = 0
    feasible_actions = 0
    adversary_patterns = 0
    all_changed_reductions = 0
    uniform_degree_reductions = 0
    sufficient_budget_reductions = 0
    for degrees in itertools.product(range(4), repeat=n):
        for changed in itertools.product((0, 1), repeat=n):
            allowed_patterns = 2 ** sum(changed)
            for budget in range(n + 1):
                instances += 1
                action_errors = []
                for selected in subsets_with_budget(n, budget):
                    feasible_actions += 1
                    adversary_patterns += allowed_patterns
                    enumerated = adversarial_error(degrees, changed, selected)
                    closed_form = formula_error(degrees, changed, selected)
                    if enumerated != closed_form:
                        raise AssertionError(
                            ("worst-case identity", degrees, changed, budget, selected,
                             enumerated, closed_form)
                        )
                    action_errors.append(enumerated)
                optimum = min(action_errors)
                selected = our_selection(degrees, changed, budget)
                achieved = adversarial_error(degrees, changed, selected)
                if achieved != optimum:
                    raise AssertionError(
                        ("top-degree optimum", degrees, changed, budget, selected,
                         achieved, optimum)
                    )

                if all(changed):
                    all_changed_reductions += 1
                    if selected != degree_selection(degrees, budget):
                        raise AssertionError(
                            ("all-changed reduction", degrees, changed, budget,
                             selected, degree_selection(degrees, budget))
                        )

                changed_degrees = [degrees[i] for i, value in enumerate(changed) if value]
                if changed_degrees and len(set(changed_degrees)) == 1:
                    uniform_degree_reductions += 1
                    changed_id = changed_id_selection(changed, budget)
                    if adversarial_error(degrees, changed, changed_id) != achieved:
                        raise AssertionError(
                            ("uniform-degree reduction", degrees, changed, budget,
                             selected, changed_id, achieved,
                             adversarial_error(degrees, changed, changed_id))
                        )

                if budget >= sum(changed):
                    sufficient_budget_reductions += 1
                    if achieved != 0:
                        raise AssertionError(
                            ("sufficient-budget reduction", degrees, changed,
                             budget, selected, achieved)
                        )

    payload = {
        "status": "VERIFIED",
        "scope": {
            "semantic_cells": n,
            "degree_values": [0, 1, 2, 3],
            "input_change_patterns": 2 ** n,
            "budgets": list(range(n + 1)),
        },
        "instances": instances,
        "feasible_actions_checked": feasible_actions,
        "adversary_patterns_checked": adversary_patterns,
        "checks": {
            "worst_case_identity": True,
            "top_degree_minimax_optimality": True,
            "all_changed_equals_degree_greedy": True,
            "uniform_changed_degrees_equal_changed_first": True,
            "sufficient_budget_zero_residual": True,
        },
        "ordinary_regime_instances": {
            "all_changed": all_changed_reductions,
            "uniform_changed_degrees": uniform_degree_reductions,
            "sufficient_budget": sufficient_budget_reductions,
        },
        "boundary": "Finite exhaustive verification supports but does not replace the general exchange proof.",
        "script_sha256_before_output": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

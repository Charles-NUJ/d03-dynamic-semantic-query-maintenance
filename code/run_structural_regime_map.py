import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist


METHODS = (
    "random_all",
    "random_changed",
    "input_change_id",
    "degree_all",
    "signature_degree_ours",
    "hindsight_oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def coefficient_of_variation(values):
    center = mean(values)
    return population_std(values) / center


def integer_degree_template(page_count: int, total_degree: int, target_cv: float):
    if target_cv == 0:
        if total_degree % page_count:
            raise ValueError("Uniform degree requires an exactly divisible total.")
        return [total_degree // page_count] * page_count

    normal = NormalDist()
    quantiles = [
        normal.inv_cdf((rank + 0.5) / page_count) for rank in range(page_count)
    ]

    def degrees_for_sigma(sigma):
        weights = [math.exp(sigma * quantile) for quantile in quantiles]
        remaining = total_degree - page_count
        scaled = [value * remaining / sum(weights) for value in weights]
        degrees = [1 + math.floor(value) for value in scaled]
        remainder = total_degree - sum(degrees)
        order = sorted(
            range(page_count),
            key=lambda index: (-(scaled[index] - math.floor(scaled[index])), index),
        )
        for index in order[:remainder]:
            degrees[index] += 1
        return degrees

    lower = 0.0
    upper = 5.0
    degrees = degrees_for_sigma(upper)
    for _ in range(80):
        sigma = (lower + upper) / 2
        candidate = degrees_for_sigma(sigma)
        achieved = coefficient_of_variation(candidate)
        degrees = candidate
        if achieved < target_cv:
            lower = sigma
        else:
            upper = sigma
    if min(degrees) <= 0 or sum(degrees) != total_degree:
        raise ValueError("Degree template construction failed.")
    return degrees


def load_cells(database_path: Path):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT o.pageid, o.predicate_index, o.old_label, n.new_label,
               s.input_changed
        FROM semantic_cell_old AS o
        JOIN oracle_semantic_cell_new AS n
          USING (pageid, predicate_index)
        JOIN semantic_input_state AS s
          USING (pageid, predicate_index)
        ORDER BY o.pageid, o.predicate_index
        """
    ).fetchall()
    reference_rows = connection.execute(
        "SELECT COUNT(*) AS count FROM reference_new"
    ).fetchone()["count"]
    connection.close()

    cells = {}
    for row in rows:
        key = (int(row["pageid"]), int(row["predicate_index"]))
        cells[key] = {
            "pageid": key[0],
            "old_label": int(row["old_label"]),
            "new_label": int(row["new_label"]),
            "observed_input_changed": int(row["input_changed"]),
            "transition": abs(int(row["new_label"]) - int(row["old_label"])),
        }
    return cells, reference_rows


def top_by_score(candidates, budget, score):
    ordered = sorted(candidates, key=lambda key: (-score[key], key[0], key[1]))
    return set(ordered[: min(budget, len(ordered))])


def select_methods(all_keys, changed, transition, degree, budget, seed):
    rng = random.Random(seed)
    changed_list = sorted(changed)
    all_list = sorted(all_keys)
    degree_score = {key: degree[key] for key in all_keys}
    hindsight_score = {key: degree[key] * transition[key] for key in all_keys}
    return {
        "random_all": set(rng.sample(all_list, min(budget, len(all_list)))),
        "random_changed": set(
            rng.sample(changed_list, min(budget, len(changed_list)))
        ),
        "input_change_id": set(changed_list[: min(budget, len(changed_list))]),
        "degree_all": top_by_score(all_keys, budget, degree_score),
        "signature_degree_ours": top_by_score(changed, budget, degree_score),
        "hindsight_oracle": top_by_score(all_keys, budget, hindsight_score),
    }


def evaluate(selected, changed, cells, degree):
    minimax_denominator = sum(degree[key] for key in changed)
    join_denominator = sum(
        degree[key] * cell["new_label"] for key, cell in cells.items()
    )
    if minimax_denominator <= 0 or join_denominator <= 0:
        raise ValueError("Every denominator must be positive.")
    minimax_residual = sum(degree[key] for key in changed if key not in selected)
    realized_error = sum(
        degree[key] * cells[key]["transition"]
        for key in cells
        if key not in selected
    )
    return {
        "calls": len(selected),
        "stable_calls": len(selected - changed),
        "minimax_residual_fraction": minimax_residual / minimax_denominator,
        "realized_join_error_fraction": realized_error / join_denominator,
    }


def lookup(rows, changed_fraction, target_cv, budget_fraction, method):
    return next(
        row
        for row in rows
        if abs(row["changed_fraction"] - changed_fraction) < 1e-12
        and abs(row["target_degree_cv"] - target_cv) < 1e-12
        and abs(row["budget_fraction_of_changed"] - budget_fraction) < 1e-12
        and row["method"] == method
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()

    cells, observed_reference_rows = load_cells(args.database)
    all_keys = sorted(cells)
    pages = sorted({key[0] for key in all_keys})
    transition_keys = {key for key in all_keys if cells[key]["transition"]}
    stable_transition_violation = any(
        cells[key]["transition"] and not cells[key]["observed_input_changed"]
        for key in all_keys
    )
    if stable_transition_violation:
        raise ValueError("The source trace violates exact input reuse.")

    changed_fractions = (0.10, 0.25, 145 / 320, 1.0)
    target_cvs = (0.0, 0.5, 1.0, 1.88)
    budget_fractions = (0.10, 0.25, 0.55, 1.0)
    total_degree = len(pages) * round(observed_reference_rows / len(pages))
    templates = {
        target_cv: integer_degree_template(len(pages), total_degree, target_cv)
        for target_cv in target_cvs
    }
    achieved_cvs = {
        target_cv: coefficient_of_variation(template)
        for target_cv, template in templates.items()
    }

    accumulators = defaultdict(lambda: defaultdict(list))
    nontransition_keys = [key for key in all_keys if key not in transition_keys]
    for repetition in range(args.repetitions):
        for target_cv in target_cvs:
            page_degrees = list(templates[target_cv])
            random.Random(2026110200 + repetition * 101 + int(target_cv * 100)).shuffle(
                page_degrees
            )
            degree_by_page = dict(zip(pages, page_degrees))
            degree = {key: degree_by_page[key[0]] for key in all_keys}

            for fraction_index, changed_fraction in enumerate(changed_fractions):
                changed_count = max(
                    len(transition_keys), round(changed_fraction * len(all_keys))
                )
                extra_count = changed_count - len(transition_keys)
                changed_rng = random.Random(
                    202611020000
                    + repetition * 10007
                    + int(target_cv * 100) * 101
                    + fraction_index
                )
                changed = set(transition_keys)
                changed.update(changed_rng.sample(nontransition_keys, extra_count))

                transition = {key: cells[key]["transition"] for key in all_keys}
                for budget_fraction in budget_fractions:
                    budget = min(
                        changed_count,
                        max(1, round(budget_fraction * changed_count)),
                    )
                    method_seed = (
                        20261102000000
                        + repetition * 1000003
                        + int(target_cv * 100) * 1009
                        + fraction_index * 101
                        + int(budget_fraction * 100)
                    )
                    selections = select_methods(
                        all_keys, changed, transition, degree, budget, method_seed
                    )
                    for method, selected in selections.items():
                        key = (changed_fraction, target_cv, budget_fraction, method)
                        outcome = evaluate(selected, changed, cells, degree)
                        for name, value in outcome.items():
                            accumulators[key][name].append(value)

    rows = []
    for changed_fraction in changed_fractions:
        changed_count = max(
            len(transition_keys), round(changed_fraction * len(all_keys))
        )
        for target_cv in target_cvs:
            for budget_fraction in budget_fractions:
                budget = min(
                    changed_count,
                    max(1, round(budget_fraction * changed_count)),
                )
                for method in METHODS:
                    values = accumulators[
                        (changed_fraction, target_cv, budget_fraction, method)
                    ]
                    rows.append(
                        {
                            "changed_fraction": changed_fraction,
                            "changed_count": changed_count,
                            "target_degree_cv": target_cv,
                            "achieved_degree_cv": achieved_cvs[target_cv],
                            "budget_fraction_of_changed": budget_fraction,
                            "budget": budget,
                            "method": method,
                            "repetitions": args.repetitions,
                            "mean_calls": mean(values["calls"]),
                            "mean_stable_calls": mean(values["stable_calls"]),
                            "mean_minimax_residual_fraction": mean(
                                values["minimax_residual_fraction"]
                            ),
                            "std_minimax_residual_fraction": population_std(
                                values["minimax_residual_fraction"]
                            ),
                            "mean_realized_join_error_fraction": mean(
                                values["realized_join_error_fraction"]
                            ),
                            "std_realized_join_error_fraction": population_std(
                                values["realized_join_error_fraction"]
                            ),
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "structural_regime_map.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    observed_fraction = 145 / 320
    special_low_budget = {
        method: lookup(rows, observed_fraction, 1.88, 0.25, method)
        for method in ("random_changed", "degree_all", "signature_degree_ours")
    }
    special_observed_budget = {
        method: lookup(rows, observed_fraction, 1.88, 0.55, method)
        for method in ("random_changed", "degree_all", "signature_degree_ours")
    }
    checks = {
        "expected_row_count": len(rows)
        == len(changed_fractions)
        * len(target_cvs)
        * len(budget_fractions)
        * len(METHODS),
        "all_repetitions_present": all(
            row["repetitions"] == args.repetitions for row in rows
        ),
        "all_changed_equals_degree_all": all(
            abs(
                lookup(rows, 1.0, cv, budget, "signature_degree_ours")[
                    "mean_minimax_residual_fraction"
                ]
                - lookup(rows, 1.0, cv, budget, "degree_all")[
                    "mean_minimax_residual_fraction"
                ]
            )
            < 1e-12
            and abs(
                lookup(rows, 1.0, cv, budget, "signature_degree_ours")[
                    "mean_realized_join_error_fraction"
                ]
                - lookup(rows, 1.0, cv, budget, "degree_all")[
                    "mean_realized_join_error_fraction"
                ]
            )
            < 1e-12
            for cv in target_cvs
            for budget in budget_fractions
        ),
        "uniform_degree_equals_changed_id_minimax": all(
            abs(
                lookup(rows, fraction, 0.0, budget, "signature_degree_ours")[
                    "mean_minimax_residual_fraction"
                ]
                - lookup(rows, fraction, 0.0, budget, "input_change_id")[
                    "mean_minimax_residual_fraction"
                ]
            )
            < 1e-12
            for fraction in changed_fractions
            for budget in budget_fractions
        ),
        "sufficient_budget_zero_for_changed_first": all(
            abs(
                lookup(rows, fraction, cv, 1.0, method)[
                    "mean_minimax_residual_fraction"
                ]
            )
            < 1e-12
            and abs(
                lookup(rows, fraction, cv, 1.0, method)[
                    "mean_realized_join_error_fraction"
                ]
            )
            < 1e-12
            for fraction in changed_fractions
            for cv in target_cvs
            for method in (
                "random_changed",
                "input_change_id",
                "signature_degree_ours",
            )
        ),
        "special_structure_improves_minimax_over_single_controls": (
            special_observed_budget["signature_degree_ours"][
                "mean_minimax_residual_fraction"
            ]
            < special_observed_budget["random_changed"][
                "mean_minimax_residual_fraction"
            ]
            and special_observed_budget["signature_degree_ours"][
                "mean_minimax_residual_fraction"
            ]
            < special_observed_budget["degree_all"][
                "mean_minimax_residual_fraction"
            ]
        ),
        "special_structure_improves_realized_error_over_single_controls": (
            special_observed_budget["signature_degree_ours"][
                "mean_realized_join_error_fraction"
            ]
            < special_observed_budget["random_changed"][
                "mean_realized_join_error_fraction"
            ]
            and special_observed_budget["signature_degree_ours"][
                "mean_realized_join_error_fraction"
            ]
            < special_observed_budget["degree_all"][
                "mean_realized_join_error_fraction"
            ]
        ),
        "selection_respects_call_cap": all(
            row["mean_calls"] <= row["budget"] + 1e-12 for row in rows
        ),
        "positive_generated_degrees": all(
            min(template) > 0 for template in templates.values()
        ),
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    payload = {
        "status": status,
        "study": (
            "Controlled structural regime map over changed-input fraction, "
            "join-degree heterogeneity, and budget coverage."
        ),
        "causal_contract": (
            "Every policy sees the generated changed set, generated current degrees, "
            "and budget before selection. Realized transitions remain hidden."
        ),
        "controlled_construction": (
            "The semantic labels and transition cells are frozen from the natural trace. "
            "Every generated changed set contains all transition cells. Degree multisets "
            "are assigned at document level, independently of transitions, with fixed "
            "total relation mass."
        ),
        "counts": {
            "cells": len(all_keys),
            "documents": len(pages),
            "transition_cells": len(transition_keys),
            "observed_reference_rows": observed_reference_rows,
            "controlled_reference_rows": total_degree,
            "summary_rows": len(rows),
            "repetitions_per_cell": args.repetitions,
        },
        "factors": {
            "changed_fractions": changed_fractions,
            "target_degree_cvs": target_cvs,
            "achieved_degree_cvs": achieved_cvs,
            "budget_fractions_of_changed": budget_fractions,
            "methods": METHODS,
        },
        "checks": checks,
        "special_structure_low_budget_cell": special_low_budget,
        "special_structure_observed_budget_cell": special_observed_budget,
        "ordinary_boundaries": {
            "all_inputs_changed": "ours equals degree-only selection",
            "uniform_join_degree": "ours equals changed-input/ID in minimax residual",
            "budget_covers_changed_layer": "every changed-first rule reaches zero",
        },
        "evidence_boundary": (
            "This controlled study explains structural activation using one frozen set "
            "of semantic outcomes. It does not estimate production occurrence frequency."
        ),
        "hashes": {
            "database": sha256(args.database),
            "script": sha256(Path(__file__)),
            "summary_csv": sha256(csv_path),
        },
    }
    json_path = args.output_dir / "structural_regime_map.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

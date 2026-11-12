import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


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


def select(candidates, budget, score):
    ordered = sorted(candidates, key=lambda item: (-score[item], item[0], item[1]))
    return set(ordered[:budget])


def join_error_fraction(cells, selected, denominator):
    error = sum(
        cell["degree"] * cell["transition"]
        for key, cell in cells.items()
        if key not in selected
    )
    return error / denominator


def load_cells(database_path: Path):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    degrees = {
        row["pageid"]: row["degree"]
        for row in connection.execute(
            "SELECT pageid, COUNT(*) AS degree FROM reference_new GROUP BY pageid"
        )
    }
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
    connection.close()
    cells = {}
    for row in rows:
        key = (row["pageid"], row["predicate_index"])
        cells[key] = {
            "degree": degrees.get(row["pageid"], 0),
            "changed": int(row["input_changed"]),
            "transition": abs(int(row["new_label"]) - int(row["old_label"])),
            "new_label": int(row["new_label"]),
        }
    return cells


def expected_result(method_results_path: Path, method: str, budget: int):
    with method_results_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == method and int(row["budget"]) == budget:
                return float(row["join_row_error_fraction"])
    raise KeyError((method, budget))


def signature_coarsening_study(cells, denominator, budgets, repetitions):
    all_keys = list(cells)
    changed = [key for key in all_keys if cells[key]["changed"]]
    unchanged = [key for key in all_keys if not cells[key]["changed"]]
    degree_score = {key: cells[key]["degree"] for key in all_keys}
    rows = []
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for fraction in fractions:
        sample_size = round(fraction * len(unchanged))
        samples = 1 if fraction in (0.0, 1.0) else repetitions
        values = defaultdict(list)
        observed_count = len(changed) + sample_size
        for repetition in range(samples):
            if sample_size == 0:
                false_positives = []
            elif sample_size == len(unchanged):
                false_positives = unchanged
            else:
                seed = 20261102 + int(1000 * fraction) * 10000 + repetition
                false_positives = random.Random(seed).sample(unchanged, sample_size)
            observed_changed = changed + list(false_positives)
            id_score = {key: 0 for key in observed_changed}
            for budget in budgets:
                selections = {
                    "change-degree": select(observed_changed, budget, degree_score),
                    "change-ID": select(observed_changed, budget, id_score),
                    "link-degree": select(all_keys, budget, degree_score),
                }
                for method, chosen in selections.items():
                    values[(budget, method)].append(
                        join_error_fraction(cells, chosen, denominator)
                    )
        for budget in budgets:
            for method in ("change-degree", "change-ID", "link-degree"):
                series = values[(budget, method)]
                rows.append(
                    {
                        "false_positive_fraction": fraction,
                        "observed_changed_count": observed_count,
                        "budget": budget,
                        "budget_fraction_of_all": budget / len(all_keys),
                        "budget_fraction_of_observed_changed": min(1.0, budget / observed_count),
                        "method": method,
                        "repetitions": len(series),
                        "mean_join_error_fraction": mean(series),
                        "std_join_error_fraction": population_std(series),
                    }
                )
    return rows


def degree_resolution_study(cells, denominator, budgets):
    changed = [key for key, cell in cells.items() if cell["changed"]]
    distinct_degrees = sorted({cells[key]["degree"] for key in changed})
    degree_rank = {degree: rank for rank, degree in enumerate(distinct_degrees)}
    rows = []
    for bins in (1, 2, 4, 8, 16, "exact"):
        if bins == "exact":
            score = {key: cells[key]["degree"] for key in changed}
            label = "exact"
        else:
            score = {
                key: min(
                    bins - 1,
                    degree_rank[cells[key]["degree"]] * bins // len(distinct_degrees),
                )
                for key in changed
            }
            label = str(bins)
        for budget in budgets:
            chosen = select(changed, budget, score)
            rows.append(
                {
                    "degree_bins": label,
                    "distinct_score_values": len(set(score.values())),
                    "budget": budget,
                    "budget_fraction_of_changed": min(1.0, budget / len(changed)),
                    "join_error_fraction": join_error_fraction(cells, chosen, denominator),
                }
            )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--method-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()

    cells = load_cells(args.database)
    all_keys = list(cells)
    changed = [key for key in all_keys if cells[key]["changed"]]
    transitions = [key for key in all_keys if cells[key]["transition"]]
    denominator = sum(cell["degree"] * cell["new_label"] for cell in cells.values())
    if denominator <= 0:
        raise ValueError("The full join denominator must be positive.")
    if any(cells[key]["transition"] and not cells[key]["changed"] for key in all_keys):
        raise ValueError("A transition appears outside the exact changed-input set.")

    budgets = [32, 80, 160]
    signature_rows = signature_coarsening_study(
        cells, denominator, budgets, args.repetitions
    )
    resolution_rows = degree_resolution_study(cells, denominator, budgets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = args.output_dir / "signature_coarsening.csv"
    resolution_path = args.output_dir / "degree_resolution.csv"
    write_csv(signature_path, signature_rows)
    write_csv(resolution_path, resolution_rows)

    def observed(rows, fraction, budget, method):
        return next(
            row["mean_join_error_fraction"]
            for row in rows
            if row["false_positive_fraction"] == fraction
            and row["budget"] == budget
            and row["method"] == method
        )

    checks = {
        "base_change_degree_matches_experiment_b32": abs(
            observed(signature_rows, 0.0, 32, "change-degree")
            - expected_result(args.method_results, "change_x_influence_ours", 32)
        ) < 1e-12,
        "base_change_degree_matches_experiment_b80": abs(
            observed(signature_rows, 0.0, 80, "change-degree")
            - expected_result(args.method_results, "change_x_influence_ours", 80)
        ) < 1e-12,
        "all_affected_change_degree_equals_link_degree_b32": abs(
            observed(signature_rows, 1.0, 32, "change-degree")
            - observed(signature_rows, 1.0, 32, "link-degree")
        ) < 1e-12,
        "all_affected_change_degree_equals_link_degree_b80": abs(
            observed(signature_rows, 1.0, 80, "change-degree")
            - observed(signature_rows, 1.0, 80, "link-degree")
        ) < 1e-12,
        "sufficient_budget_zero_at_exact_signature": abs(
            observed(signature_rows, 0.0, 160, "change-degree")
        ) < 1e-12,
        "one_degree_bin_matches_change_id": all(
            abs(
                next(
                    row["join_error_fraction"]
                    for row in resolution_rows
                    if row["degree_bins"] == "1" and row["budget"] == budget
                )
                - observed(signature_rows, 0.0, budget, "change-ID")
            ) < 1e-12
            for budget in budgets
        ),
        "exact_degree_matches_change_degree": all(
            abs(
                next(
                    row["join_error_fraction"]
                    for row in resolution_rows
                    if row["degree_bins"] == "exact" and row["budget"] == budget
                )
                - observed(signature_rows, 0.0, budget, "change-degree")
            ) < 1e-12
            for budget in budgets
        ),
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    payload = {
        "status": status,
        "study_type": "controlled information-structure study",
        "control_boundary": (
            "The protocol control remains z_i. False-positive fraction and degree bins "
            "coarsen observable metadata and are experimental factors, not deployable actions."
        ),
        "counts": {
            "affected_cells": len(all_keys),
            "changed_input_cells": len(changed),
            "unchanged_input_cells": len(all_keys) - len(changed),
            "semantic_transitions": len(transitions),
            "full_join_rows": denominator,
        },
        "factors": {
            "signature_false_positive_fraction": [0.0, 0.25, 0.5, 0.75, 1.0],
            "degree_bins": [1, 2, 4, 8, 16, "exact"],
            "budgets": budgets,
            "random_repetitions": args.repetitions,
        },
        "checks": checks,
        "selected_results": {
            "exact_signature_b32_change_degree": observed(signature_rows, 0.0, 32, "change-degree"),
            "all_affected_b32_change_degree": observed(signature_rows, 1.0, 32, "change-degree"),
            "all_affected_b32_link_degree": observed(signature_rows, 1.0, 32, "link-degree"),
            "exact_signature_b80_change_degree": observed(signature_rows, 0.0, 80, "change-degree"),
            "all_affected_b80_change_degree": observed(signature_rows, 1.0, 80, "change-degree"),
            "all_affected_b80_link_degree": observed(signature_rows, 1.0, 80, "link-degree"),
        },
        "hashes": {
            "database": sha256(args.database),
            "method_results": sha256(args.method_results),
            "script": sha256(Path(__file__)),
            "signature_coarsening_csv": sha256(signature_path),
            "degree_resolution_csv": sha256(resolution_path),
        },
        "evidence_boundary": (
            "The study reuses one frozen workload. It identifies structural transitions "
            "inside that workload and does not estimate production occurrence frequency."
        ),
    }
    output_path = args.output_dir / "structural_regime_variable_study.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

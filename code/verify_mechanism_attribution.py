import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--method-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_csv = args.result_dir / "mechanism_attribution.csv"
    decomposition_csv = args.result_dir / "factorial_decomposition.csv"
    experiment_json = args.result_dir / "mechanism_attribution.json"
    rows = list(csv.DictReader(result_csv.open(encoding="utf-8")))
    decomposition = list(csv.DictReader(decomposition_csv.open(encoding="utf-8")))
    record = json.loads(experiment_json.read_text(encoding="utf-8"))
    published = list(csv.DictReader(args.method_results.open(encoding="utf-8")))

    methods = {"random_all", "random_changed", "degree_all", "change_degree"}
    budgets = {0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 145, 160}
    lookup = {(row["method"], int(row["budget"])): row for row in rows}
    old_lookup = {
        (row["method"], int(row["budget"])): float(row["join_row_error_fraction"])
        for row in published
    }
    mapping = {
        "random_all": "uniform_clean",
        "random_changed": "change_random_control",
        "degree_all": "influence_only",
        "change_degree": "change_x_influence_ours",
    }

    matches = []
    for budget in (32, 80):
        for method, old_method in mapping.items():
            observed = float(lookup[(method, budget)]["join_row_error_fraction"])
            expected = old_lookup[(old_method, budget)]
            matches.append(abs(observed - expected) < 0.004)

    checks = {
        "experiment_status_verified": record["status"] == "VERIFIED",
        "complete_factor_grid": len(rows) == len(methods) * len(budgets)
        and set((row["method"], int(row["budget"])) for row in rows)
        == set((method, budget) for method in methods for budget in budgets),
        "complete_decomposition_grid": len(decomposition) == len(budgets)
        and {int(row["budget"]) for row in decomposition} == budgets,
        "output_hashes_match_record": (
            sha256(result_csv) == record["hashes"]["mechanism_attribution_csv"]
            and sha256(decomposition_csv) == record["hashes"]["factorial_decomposition_csv"]
        ),
        "published_budget_rows_reproduced": all(matches),
        "gate_has_zero_stable_calls_through_threshold": all(
            abs(float(lookup[(method, budget)]["stable_calls"])) < 1e-12
            for method in ("random_changed", "change_degree")
            for budget in budgets if budget <= 145
        ),
        "ours_maximizes_changed_degree_coverage": all(
            float(lookup[("change_degree", budget)]["changed_degree_coverage"]) + 1e-12
            >= float(lookup[(method, budget)]["changed_degree_coverage"])
            for method in methods for budget in budgets
        ),
        "ours_minimizes_realized_join_error": all(
            float(lookup[("change_degree", budget)]["join_row_error_fraction"])
            <= float(lookup[(method, budget)]["join_row_error_fraction"]) + 1e-12
            for method in methods for budget in budgets
        ),
        "threshold_boundary_holds": (
            abs(float(lookup[("random_changed", 145)]["join_row_error_fraction"])) < 1e-12
            and abs(float(lookup[("change_degree", 145)]["join_row_error_fraction"])) < 1e-12
            and float(lookup[("degree_all", 145)]["join_row_error_fraction"]) > 0
        ),
        "gated_methods_stop_at_changed_set": all(
            int(float(lookup[(method, budget)]["selected"])) == min(budget, 145)
            for method in ("random_changed", "change_degree")
            for budget in budgets
        ),
        "all_methods_respect_budget_cap": all(
            int(float(row["selected"])) <= int(row["budget"])
            for row in rows
        ),
        "factorial_terms_recompute": all(
            abs(
                float(row["combined_gain_pp"])
                - (float(row["uniform_error_pct"]) - float(row["both_error_pct"]))
            ) < 1e-10
            and abs(
                float(row["overlap_pp"])
                - (
                    float(row["gate_gain_without_degree_pp"])
                    + float(row["degree_gain_without_gate_pp"])
                    - float(row["combined_gain_pp"])
                )
            ) < 1e-10
            for row in decomposition
        ),
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    output = {
        "status": status,
        "checks": checks,
        "counts": {
            "mechanism_rows": len(rows),
            "decomposition_rows": len(decomposition),
            "published_matches": sum(matches),
            "published_match_total": len(matches),
        },
        "hashes": {
            "mechanism_attribution_csv": sha256(result_csv),
            "factorial_decomposition_csv": sha256(decomposition_csv),
            "experiment_json": sha256(experiment_json),
            "method_results": sha256(args.method_results),
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

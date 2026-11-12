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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def find(rows, changed_fraction, degree_cv, budget_fraction, method):
    return next(
        row
        for row in rows
        if abs(float(row["changed_fraction"]) - changed_fraction) < 1e-12
        and abs(float(row["target_degree_cv"]) - degree_cv) < 1e-12
        and abs(float(row["budget_fraction_of_changed"]) - budget_fraction)
        < 1e-12
        and row["method"] == method
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--reproduced-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_json_path = args.primary_dir / "structural_regime_map.json"
    reproduced_json_path = args.reproduced_dir / "structural_regime_map.json"
    primary_csv_path = args.primary_dir / "structural_regime_map.csv"
    reproduced_csv_path = args.reproduced_dir / "structural_regime_map.csv"
    primary = load_json(primary_json_path)
    reproduced = load_json(reproduced_json_path)
    primary_rows = load_csv(primary_csv_path)
    reproduced_rows = load_csv(reproduced_csv_path)

    changed_fractions = (0.10, 0.25, 145 / 320, 1.0)
    degree_cvs = (0.0, 0.5, 1.0, 1.88)
    budget_fractions = (0.10, 0.25, 0.55, 1.0)
    all_changed_equal = all(
        abs(
            float(
                find(primary_rows, 1.0, cv, budget, "signature_degree_ours")[
                    metric
                ]
            )
            - float(find(primary_rows, 1.0, cv, budget, "degree_all")[metric])
        )
        < 1e-12
        for cv in degree_cvs
        for budget in budget_fractions
        for metric in (
            "mean_minimax_residual_fraction",
            "mean_realized_join_error_fraction",
        )
    )
    uniform_equal = all(
        abs(
            float(
                find(
                    primary_rows,
                    fraction,
                    0.0,
                    budget,
                    "signature_degree_ours",
                )["mean_minimax_residual_fraction"]
            )
            - float(
                find(
                    primary_rows, fraction, 0.0, budget, "input_change_id"
                )["mean_minimax_residual_fraction"]
            )
        )
        < 1e-12
        for fraction in changed_fractions
        for budget in budget_fractions
    )
    full_budget_zero = all(
        abs(
            float(
                find(primary_rows, fraction, cv, 1.0, method)[
                    "mean_minimax_residual_fraction"
                ]
            )
        )
        < 1e-12
        and abs(
            float(
                find(primary_rows, fraction, cv, 1.0, method)[
                    "mean_realized_join_error_fraction"
                ]
            )
        )
        < 1e-12
        for fraction in changed_fractions
        for cv in degree_cvs
        for method in (
            "random_changed",
            "input_change_id",
            "signature_degree_ours",
        )
    )
    special_ours = find(
        primary_rows, 145 / 320, 1.88, 0.55, "signature_degree_ours"
    )
    special_random = find(
        primary_rows, 145 / 320, 1.88, 0.55, "random_changed"
    )
    special_degree = find(primary_rows, 145 / 320, 1.88, 0.55, "degree_all")
    special_improves = all(
        float(special_ours[metric]) < float(special_random[metric])
        and float(special_ours[metric]) < float(special_degree[metric])
        for metric in (
            "mean_minimax_residual_fraction",
            "mean_realized_join_error_fraction",
        )
    )

    checks = {
        "primary_status_verified": primary.get("status") == "VERIFIED",
        "reproduced_status_verified": reproduced.get("status") == "VERIFIED",
        "primary_internal_checks_pass": all(primary.get("checks", {}).values()),
        "reproduced_internal_checks_pass": all(
            reproduced.get("checks", {}).values()
        ),
        "summary_hashes_match": sha256(primary_csv_path)
        == sha256(reproduced_csv_path),
        "summary_rows_match": primary_rows == reproduced_rows,
        "expected_summary_rows": len(primary_rows) == 384,
        "two_hundred_repetitions_each": all(
            int(row["repetitions"]) == 200 for row in primary_rows
        ),
        "database_hashes_match": primary["hashes"]["database"]
        == reproduced["hashes"]["database"],
        "script_hashes_match": primary["hashes"]["script"]
        == reproduced["hashes"]["script"],
        "ordinary_boundaries_recorded": set(primary["ordinary_boundaries"])
        == {
            "all_inputs_changed",
            "uniform_join_degree",
            "budget_covers_changed_layer",
        },
        "all_changed_equality_recomputed": all_changed_equal,
        "uniform_degree_equality_recomputed": uniform_equal,
        "full_budget_zero_recomputed": full_budget_zero,
        "special_structure_gain_recomputed": special_improves,
        "ours_never_spends_on_stable_cells": all(
            abs(float(row["mean_stable_calls"])) < 1e-12
            for row in primary_rows
            if row["method"] == "signature_degree_ours"
        ),
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    record = {
        "status": status,
        "checks": checks,
        "primary": {
            "json": str(primary_json_path),
            "json_sha256": sha256(primary_json_path),
            "csv": str(primary_csv_path),
            "csv_sha256": sha256(primary_csv_path),
        },
        "reproduced": {
            "json": str(reproduced_json_path),
            "json_sha256": sha256(reproduced_json_path),
            "csv": str(reproduced_csv_path),
            "csv_sha256": sha256(reproduced_csv_path),
        },
        "boundary": (
            "This verifier checks deterministic reproduction and registered structural "
            "equalities. It does not validate production frequency or user demand."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

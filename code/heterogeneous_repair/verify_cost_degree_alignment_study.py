"""Independent record-level checks for the cost--degree alignment study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    records = payload["records"]

    checks = {
        "producer_status_verified": payload["status"] == "VERIFIED",
        "candidate_count_145": payload["candidate_cells"] == 145,
        "four_pairings": len(payload["pairing_summaries"]) == 4,
        "four_budget_fractions": payload["budget_fractions"] == [0.2, 0.4, 0.6, 0.8],
        "five_methods_each_cell": len(records) == 4 * 4 * 5,
        "csv_json_row_count_match": len(csv_rows) == len(records),
        "all_dp_milp_checks_match": payload["dp_milp_checks"] == 16
        and payload["dp_milp_mismatches"] == 0,
        "exact_gap_zero": all(
            abs(float(row["optimality_gap_fraction"])) <= 1e-15
            for row in records
            if row["method"] == "signature_knapsack_dp_ours"
        ),
        "all_actions_feasible": all(
            float(row["used_cost"]) <= int(row["budget"]) + 1e-9
            for row in records
        ),
        "all_gaps_nonnegative": all(
            float(row["optimality_gap_fraction"]) >= -1e-12 for row in records
        ),
        "random_has_1000_repetitions": all(
            int(row["repetitions"]) == 1000
            for row in records
            if row["method"] == "random_control_ours"
        ),
    }
    observed = next(
        row for row in payload["pairing_summaries"] if row["pairing"] == "observed"
    )
    checks["observed_positive_rank_correlation"] = (
        0.74 < float(observed["spearman_degree_cost"]) < 0.75
    )
    checks["observed_density_max_gap_below_half_percent"] = (
        float(observed["degree_per_cost_ours_max_gap"]) < 0.005
    )
    checks["observed_degree_max_gap_below_one_percent"] = (
        float(observed["largest_degree_ours_max_gap"]) < 0.01
    )

    result = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "json_sha256": sha256(args.json),
        "csv_sha256": sha256(args.csv),
        "record_count": len(records),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

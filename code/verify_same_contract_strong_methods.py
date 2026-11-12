from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_number(rows, field: str, tolerance: float = 1e-12) -> bool:
    values = [float(row[field]) for row in rows]
    return max(values) - min(values) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    csv_path = result_dir / "same_contract_strong_methods.csv"
    record_path = result_dir / "execution_record.json"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    execution = json.loads(record_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["window"], int(row["budget"]))].append(row)
    expected_methods = {
        "signature_degree_greedy_ours",
        "central_exact_dp_ours",
        "dual_price_iteration_ours",
    }
    group_checks = {}
    for key, group in sorted(grouped.items()):
        methods = {row["method"] for row in group}
        group_checks[f"{key[0]}_b{key[1]}"] = {
            "three_methods": methods == expected_methods,
            "same_selected_set": len({row["selection_sha256"] for row in group}) == 1,
            "same_selected_weight": same_number(group, "selected_weight"),
            "same_residual_weight": same_number(group, "residual_weight"),
            "same_filter_error": same_number(group, "filter_error_fraction"),
            "same_join_error": same_number(group, "join_row_error_fraction"),
            "same_group_error": same_number(group, "group_l1_error_fraction"),
            "dual_iterated": next(int(row["iterations"]) for row in group if row["method"] == "dual_price_iteration_ours") > 0,
        }
    checks = {
        "execution_finished": execution.get("status") == "FINISHED",
        "twelve_rows": len(rows) == 12,
        "two_windows_two_budgets": len(grouped) == 4,
        "result_hash_matches_record": execution["result"]["sha256"] == sha256(csv_path),
        "all_group_checks_pass": all(all(values.values()) for values in group_checks.values()),
        "positive_external_denominators": all(
            min(
                window["full_filter_mass"],
                window["full_join_mass"],
                window["full_group_mass"],
            ) > 0
            for window in execution["windows"].values()
        ),
    }
    output = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "group_checks": group_checks,
        "hashes": {
            "result_csv": sha256(csv_path),
            "execution_record": sha256(record_path),
            "verifier": sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

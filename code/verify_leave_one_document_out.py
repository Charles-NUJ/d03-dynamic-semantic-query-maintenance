import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_METHODS = {
    "signature_degree_ours",
    "input_change_id",
    "link_degree",
    "proxy_degree",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def compare(values, tolerance=1e-12):
    return {
        "wins": sum(value > tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
        "losses": sum(value < -tolerance for value in values),
    }


def close(left, right, tolerance=1e-10):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def summarize(rows):
    by_method = defaultdict(list)
    by_page = defaultdict(dict)
    for row in rows:
        method = row["method"]
        value = 100.0 * float(row["join_row_error_fraction"])
        by_method[method].append(value)
        by_page[int(row["omitted_pageid"])][method] = value
    controls = ["input_change_id", "link_degree", "proxy_degree"]
    pairwise = {}
    for control in controls:
        gains = [
            values[control] - values["signature_degree_ours"]
            for _, values in sorted(by_page.items())
        ]
        pairwise[control] = {"gain_pp": distribution(gains), **compare(gains)}
    best_gains = [
        min(values[control] for control in controls) - values["signature_degree_ours"]
        for _, values in sorted(by_page.items())
    ]
    return {
        "join_error_percent": {
            method: distribution(values) for method, values in sorted(by_method.items())
        },
        "ours_against_controls": pairwise,
        "ours_against_best_control": {"gain_pp": distribution(best_gains), **compare(best_gains)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--reproduced-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_csv = args.primary_dir / "leave_one_document_out.csv"
    primary_json = args.primary_dir / "leave_one_document_out.json"
    reproduced_csv = args.reproduced_dir / "leave_one_document_out.csv"
    reproduced_json = args.reproduced_dir / "leave_one_document_out.json"
    with primary_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(primary_json.read_text(encoding="utf-8"))
    pages = defaultdict(list)
    for row in rows:
        pages[int(row["omitted_pageid"])].append(row)

    recomputed = summarize(rows)
    input_paths = {
        "pairs": args.artifact_dir / "inputs/frozen_pairs.json",
        "revisions": args.artifact_dir / "inputs/revisions.json",
        "pair_results": args.artifact_dir / "results/pair_predicate_results.json",
        "predictions": args.artifact_dir / "results/predictions.json",
    }
    checks = {
        "primary_status_finished": summary.get("status") == "FINISHED",
        "primary_reproduced_csv_identical": sha256(primary_csv) == sha256(reproduced_csv),
        "primary_reproduced_json_identical": sha256(primary_json) == sha256(reproduced_json),
        "sixty_four_documents": len(pages) == 64,
        "four_methods_per_document": all(len(items) == 4 for items in pages.values()),
        "expected_methods": all({item["method"] for item in items} == EXPECTED_METHODS for items in pages.values()),
        "fixed_budget": all(int(row["budget"]) == 80 for row in rows),
        "positive_denominators": all(
            min(int(row["full_filter_mass"]), int(row["full_join_mass"]), int(row["full_group_mass"])) > 0
            for row in rows
        ),
        "selection_respects_cap": all(int(row["selected"]) <= int(row["budget"]) for row in rows),
        "summary_recomputed": close(
            recomputed,
            {
                "join_error_percent": summary["join_error_percent"],
                "ours_against_controls": summary["ours_against_controls"],
                "ours_against_best_control": summary["ours_against_best_control"],
            },
        ),
        "input_hashes_match": all(
            summary["input_sha256"][name] == sha256(path) for name, path in input_paths.items()
        ),
        "sensitivity_boundary_registered": "not 64 independent" in summary.get("boundary", ""),
    }
    record = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "counts": {"documents": len(pages), "rows": len(rows)},
        "hashes": {
            "primary_csv": sha256(primary_csv),
            "primary_json": sha256(primary_json),
            "reproduced_csv": sha256(reproduced_csv),
            "reproduced_json": sha256(reproduced_json),
        },
        "boundary": "This verifier establishes deterministic omission sensitivity and summary arithmetic. It does not establish independent replication, production frequency, or business importance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if record["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

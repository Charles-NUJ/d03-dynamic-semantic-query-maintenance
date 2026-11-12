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


def main() -> None:
    artifact_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=artifact_dir / "results/result_mechanism_diagnostics",
    )
    parser.add_argument(
        "--reproduced-dir",
        type=Path,
        default=artifact_dir / "reproduced_results/result_mechanism_diagnostics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "results/result_mechanism_diagnostics/verification.json",
    )
    args = parser.parse_args()
    primary_json = args.result_dir / "result_mechanism_diagnostics.json"
    reproduced_json = args.reproduced_dir / "result_mechanism_diagnostics.json"
    primary_csv = args.result_dir / "result_mechanism_diagnostics.csv"
    reproduced_csv = args.reproduced_dir / "result_mechanism_diagnostics.csv"
    payload = json.loads(primary_json.read_text(encoding="utf-8"))
    records = payload["records"]
    checks = {}

    checks["primary_and_reproduced_json_identical"] = (
        primary_json.read_bytes() == reproduced_json.read_bytes()
    )
    checks["primary_and_reproduced_csv_identical"] = (
        primary_csv.read_bytes() == reproduced_csv.read_bytes()
    )
    checks["twenty_deterministic_cells"] = len(records) == 20
    checks["positive_recorded_denominators"] = (
        payload["totals"]["changed_degree"] > 0
        and payload["totals"]["transition_degree"] > 0
        and payload["totals"]["transitions"] > 0
    )
    checks["error_numerators_match_missed_transitions"] = all(
        item["filter_error_numerator"] == item["transition_missed"]
        and item["join_error_numerator"] == item["transition_degree_missed"]
        for item in records
    )
    checks["our_rule_uses_no_stable_calls"] = all(
        item["stable_selected"] == 0
        for item in records
        if item["method"] == "signature_degree_ours"
    )
    checks["our_and_input_change_close_at_changed_count"] = all(
        item["filter_error_numerator"] == 0 and item["join_error_numerator"] == 0
        for item in records
        if item["method"] in {"signature_degree_ours", "input_change_id"}
        and item["budget"] in {145, 160}
    )
    flat = payload["explanations"]["link_degree_flat_32_to_80"]
    checks["link_degree_flatness_explained"] = (
        flat["additional_calls"] == 48 and flat["additional_transition_cells"] == 0
    )
    tie = payload["explanations"]["filter_tie_at_32"]
    checks["filter_tie_and_join_difference_explained"] = (
        tie["our_transition_selected"] == tie["proxy_transition_selected"]
        and tie["our_transition_degree_selected"] > tie["proxy_transition_degree_selected"]
    )
    with primary_csv.open("r", encoding="utf-8", newline="") as stream:
        checks["csv_reopens_with_twenty_rows"] = len(list(csv.DictReader(stream))) == 20

    status = "VERIFIED" if all(checks.values()) else "FAILED"
    output = {
        "status": status,
        "checks": checks,
        "evidence": {
            "primary_json_sha256": sha256(primary_json),
            "reproduced_json_sha256": sha256(reproduced_json),
            "primary_csv_sha256": sha256(primary_csv),
            "reproduced_csv_sha256": sha256(reproduced_csv),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

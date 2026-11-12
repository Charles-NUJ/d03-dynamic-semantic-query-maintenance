import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tol: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tol, abs_tol=tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = args.package.resolve()
    artifact = package / "artifact" if (package / "artifact").is_dir() else package
    record_path = args.record.resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    inputs = artifact / "inputs"
    results = artifact / "results"
    paths = {
        "pairs": inputs / "frozen_pairs.json",
        "revisions": inputs / "revisions.json",
        "pair_results": results / "pair_predicate_results.json",
        "predictions": results / "predictions.json",
    }
    pairs = json.loads(paths["pairs"].read_text(encoding="utf-8"))["pairs"]
    ordered = sorted(
        pairs,
        key=lambda item: (
            datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
            int(item["pageid"]),
        ),
    )
    expected_pageids = {
        "early_32": [int(item["pageid"]) for item in ordered[:32]],
        "late_32": [int(item["pageid"]) for item in ordered[32:]],
    }
    early = record["splits"]["early_32"]
    late = record["splits"]["late_32"]
    early_ours = early["join_error_percent"]["signature_degree_ours"]
    early_id = early["join_error_percent"]["input_change_id"]
    late_ours = late["join_error_percent"]["signature_degree_ours"]
    late_id = late["join_error_percent"]["input_change_id"]
    early_ours_mech = early["mechanism"]["signature_degree_ours"]
    early_id_mech = early["mechanism"]["input_change_id"]
    late_ours_mech = late["mechanism"]["signature_degree_ours"]
    late_id_mech = late["mechanism"]["input_change_id"]
    all_methods = {
        row["method"] for row in record["records"]
    }
    checks = {
        "status_finished": record["status"] == "FINISHED",
        "input_hashes_match": all(
            record["input_sha256"][name] == sha256(path)
            for name, path in paths.items()
        ),
        "temporal_halves_match_source": record["split_pageids"] == expected_pageids,
        "halves_are_disjoint": set(record["split_pageids"]["early_32"]).isdisjoint(
            record["split_pageids"]["late_32"]
        ),
        "halves_cover_all_documents": (
            set(record["split_pageids"]["early_32"])
            | set(record["split_pageids"]["late_32"])
        ) == {int(item["pageid"]) for item in pairs},
        "each_half_has_32_documents_and_160_cells": all(
            split["structure"]["documents"] == 32
            and split["structure"]["cells"] == 160
            for split in (early, late)
        ),
        "each_half_uses_25_percent_call_budget": all(
            split["structure"]["budget"] == 40 for split in (early, late)
        ),
        "nine_causal_methods_recorded": len(all_methods) == 9
        and len(record["records"]) == 18,
        "gated_methods_spend_no_stable_calls": all(
            split["mechanism"][method]["stable_selected"] == 0
            for split in (early, late)
            for method in ("signature_degree_ours", "input_change_id")
        ),
        "ours_has_lower_worst_case_residual_in_both_halves": all(
            1.0 - split["mechanism"]["signature_degree_ours"]["changed_degree_coverage"]
            < 1.0 - split["mechanism"]["input_change_id"]["changed_degree_coverage"]
            for split in (early, late)
        ),
        "early_realized_counterexample_recorded": close(
            early_ours, 12.802768166089965
        ) and close(early_id, 9.419454056132256) and early_ours > early_id,
        "early_counterexample_mechanism_recorded": (
            early_ours_mech["transition_selected"] == 9
            and early_id_mech["transition_selected"] == 11
            and early_ours_mech["transition_degree_selected"] == 1250
            and early_id_mech["transition_degree_selected"] == 1338
        ),
        "late_realized_gain_recorded": close(late_ours, 0.0)
        and close(late_id, 1.9329507701600726)
        and late_ours < late_id,
        "late_gain_mechanism_recorded": (
            late_ours_mech["transition_selected"] == 5
            and late_id_mech["transition_selected"] == 3
            and late_ours_mech["transition_degree_selected"] == 1512
            and late_id_mech["transition_degree_selected"] == 1448
        ),
        "scope_does_not_claim_independence": "not independent production traces"
        in record["scope"],
    }
    try:
        record_label = str(record_path.relative_to(package))
    except ValueError:
        record_label = record_path.name
    payload = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "derived": {
            "early_worst_case_residual_percent": {
                "ours": 100.0 * (
                    1.0 - early_ours_mech["changed_degree_coverage"]
                ),
                "input_change_id": 100.0 * (
                    1.0 - early_id_mech["changed_degree_coverage"]
                ),
            },
            "late_worst_case_residual_percent": {
                "ours": 100.0 * (
                    1.0 - late_ours_mech["changed_degree_coverage"]
                ),
                "input_change_id": 100.0 * (
                    1.0 - late_id_mech["changed_degree_coverage"]
                ),
            },
            "early_realized_join_error_percent": {
                "ours": early_ours,
                "input_change_id": early_id,
            },
            "late_realized_join_error_percent": {
                "ours": late_ours,
                "input_change_id": late_id,
            },
        },
        "files": {
            "record": record_label,
            "record_sha256": sha256(record_path),
            "runner_sha256": record["script_sha256_before_output"],
            "verifier_sha256_before_output": sha256(Path(__file__)),
        },
        "boundary": (
            "This verifies a disjoint temporal-half sensitivity inside one frozen trace. "
            "It does not establish independent temporal, model, predicate, or production replication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

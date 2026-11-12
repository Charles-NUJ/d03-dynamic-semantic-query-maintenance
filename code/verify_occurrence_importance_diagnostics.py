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
        "--primary", type=Path,
        default=artifact_dir / "results/occurrence_importance",
    )
    parser.add_argument(
        "--reproduced", type=Path,
        default=artifact_dir / "reproduced_results/occurrence_importance",
    )
    args = parser.parse_args()
    primary_json = args.primary / "occurrence_importance_summary.json"
    reproduced_json = args.reproduced / "occurrence_importance_summary.json"
    primary_csv = args.primary / "document_occurrence.csv"
    reproduced_csv = args.reproduced / "document_occurrence.csv"
    payload = json.loads(primary_json.read_text(encoding="utf-8"))
    occurrence = payload["occurrence"]
    importance = payload["external_importance"]
    with primary_csv.open("r", encoding="utf-8", newline="") as stream:
        documents = list(csv.DictReader(stream))

    checks = {
        "json_byte_identical": primary_json.read_bytes() == reproduced_json.read_bytes(),
        "csv_byte_identical": primary_csv.read_bytes() == reproduced_csv.read_bytes(),
        "document_count_64": len(documents) == 64 == payload["window"]["documents"],
        "cell_partition_320": (
            occurrence["semantic_cells"] == 320
            and occurrence["stable_cells"] == 175
            and occurrence["changed_cells"] == 145
        ),
        "transition_count_23": occurrence["transition_cells"] == 23,
        "transition_documents_13": occurrence["documents_with_transitions"] == 13,
        "exact_reuse_consistent": occurrence["transitions_on_unchanged_inputs"] == 0,
        "positive_external_denominators": (
            importance["full_filter_rows"] > 0
            and importance["full_join_rows"] > 0
            and importance["full_group_mass"] > 0
        ),
        "reuse_all_is_materially_wrong": (
            importance["reuse_all"]["filter_error"] == 23
            and importance["reuse_all"]["join_row_error"] == 3095
        ),
        "changed_full_reaches_zero": (
            importance["changed_full_calls"] == 145
            and importance["changed_full"]["filter_error"] == 0
            and importance["changed_full"]["join_row_error"] == 0
        ),
        "ours_80_matches_manuscript": (
            importance["ours_80_calls"] == 80
            and importance["ours_80"]["filter_error"] == 2
            and importance["ours_80"]["join_row_error"] == 51
        ),
        "document_totals_reconcile": (
            sum(int(item["cells"]) for item in documents) == 320
            and sum(int(item["changed_cells"]) for item in documents) == 145
            and sum(int(item["transition_cells"]) for item in documents) == 23
            and sum(int(item["transition_join_rows"]) for item in documents) == 3095
        ),
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    verification = {
        "status": status,
        "checks": checks,
        "evidence": {
            "primary_json_sha256": sha256(primary_json),
            "reproduced_json_sha256": sha256(reproduced_json),
            "primary_csv_sha256": sha256(primary_csv),
            "reproduced_csv_sha256": sha256(reproduced_csv),
        },
    }
    output = args.primary / "verification.json"
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

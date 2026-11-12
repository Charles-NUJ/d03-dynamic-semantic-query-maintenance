import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir.parent / "reproduced_results/main"
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    summary_path = args.result_dir / "summary.json"
    receipt_path = args.result_dir / "execution_receipt.json"
    database_path = args.result_dir / "structured_database.sqlite"
    results_path = args.result_dir / "method_results.csv"
    source_map_path = args.result_dir / "method_source_map.json"
    reference_path = args.result_dir / "reference_relation.json"
    ablation_path = args.result_dir / "structural_ablation.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    sources = json.loads(source_map_path.read_text(encoding="utf-8"))
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    with results_path.open("r", encoding="utf-8", newline="") as stream:
        results = list(csv.DictReader(stream))
    with ablation_path.open("r", encoding="utf-8", newline="") as stream:
        ablations = list(csv.DictReader(stream))

    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    counts = {
        "document": connection.execute("SELECT COUNT(*) FROM document").fetchone()[0],
        "semantic_cell_old": connection.execute("SELECT COUNT(*) FROM semantic_cell_old").fetchone()[0],
        "semantic_input_state": connection.execute("SELECT COUNT(*) FROM semantic_input_state").fetchone()[0],
        "oracle_semantic_cell_new": connection.execute("SELECT COUNT(*) FROM oracle_semantic_cell_new").fetchone()[0],
        "reference_new": connection.execute("SELECT COUNT(*) FROM reference_new").fetchone()[0],
        "semantic_transitions": connection.execute(
            """
            SELECT COUNT(*)
            FROM semantic_cell_old AS o
            JOIN oracle_semantic_cell_new AS n USING(pageid, predicate_index)
            WHERE o.old_label <> n.new_label
            """
        ).fetchone()[0],
        "duplicate_reference_rows": connection.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT pageid, target, COUNT(*) AS c
              FROM reference_new GROUP BY pageid, target HAVING c > 1
            )
            """
        ).fetchone()[0],
        "changed_inputs": connection.execute(
            "SELECT COUNT(*) FROM semantic_input_state WHERE input_changed = 1"
        ).fetchone()[0],
        "transitions_on_unchanged_inputs": connection.execute(
            """
            SELECT COUNT(*)
            FROM semantic_cell_old AS o
            JOIN oracle_semantic_cell_new AS n USING(pageid, predicate_index)
            JOIN semantic_input_state AS s USING(pageid, predicate_index)
            WHERE s.input_changed = 0 AND o.old_label <> n.new_label
            """
        ).fetchone()[0],
    }
    connection.close()

    checks = {
        "summary_finished": summary.get("status") == "FINISHED",
        "receipt_finished": receipt.get("status") == "FINISHED",
        "documents_match": counts["document"] == summary["structure"]["pages"] == 64,
        "semantic_cells_match": counts["semantic_cell_old"] == counts["semantic_input_state"] == counts["oracle_semantic_cell_new"] == summary["structure"]["semantic_cells"] == 320,
        "reference_rows_match": counts["reference_new"] == len(references) == summary["structure"]["reference_rows"],
        "transitions_match": counts["semantic_transitions"] == summary["structure"]["semantic_transitions"] == 23,
        "changed_inputs_match": counts["changed_inputs"] == summary["structure"]["changed_input_cells"] == 145,
        "unchanged_inputs_match": summary["structure"]["unchanged_input_cells"] == 175,
        "fixed_input_invariance_observed": counts["transitions_on_unchanged_inputs"] == summary["structure"]["transitions_on_unchanged_inputs"] == 0,
        "no_duplicate_reference_rows": counts["duplicate_reference_rows"] == 0,
        "positive_filter_denominator": summary["structure"]["full_filter_mass"] > 0,
        "positive_join_denominator": summary["structure"]["full_join_mass"] > 0,
        "positive_group_denominator": summary["structure"]["full_group_mass"] > 0,
        "method_source_map_complete": all(result["method"] in sources for result in results),
        "ours_identified": sources["change_x_influence_ours"]["kind"] == "our method",
        "literature_adaptations_identified": all(
            sources[name]["kind"] == "ability-matched literature adaptation"
            for name in ["lineage_only", "uniform_clean", "proxy_priority"]
        ),
        "fair_query_aware_controls_present": all(
            name in sources for name in ["ttl_x_influence", "edit_x_influence", "proxy_x_influence"]
        ),
        "full_recompute_zero_error": all(
            float(result["filter_error"]) == 0
            and float(result["join_row_error"]) == 0
            and float(result["group_l1_error"]) == 0
            for result in results
            if result["method"] == "lineage_only"
        ),
        "result_budgets_present": sorted({int(result["budget"]) for result in results}) == [0, 32, 80, 160, 240, 320],
        "all_methods_respect_budget_cap": all(
            int(result["selected"]) <= int(result["budget"])
            for result in results
        ),
        "gated_methods_stop_at_changed_set": all(
            int(result["selected"])
            == min(int(result["budget"]), summary["structure"]["changed_input_cells"])
            for result in results
            if result["method"] in {
                "change_only_control",
                "change_random_control",
                "change_x_influence_ours",
            }
        ),
        "input_change_label_matches_manuscript": sources["change_only_control"]["display"]
        == "Input-change/ID (control)",
        "displayed_method_labels_match_manuscript": all(
            sources[name]["display"] == display
            for name, display in {
                "reuse_all": "Reuse-all (endpoint)",
                "uniform_clean": "SVC-style uniform [SVC]",
                "proxy_priority": "Proxy uncertainty [SUPG]",
                "proxy_x_influence": "Proxy-degree [SUPG]",
                "ttl_x_influence": "TTL-degree greedy (control)",
                "edit_x_influence": "Edit-degree greedy (control)",
                "influence_only": "Link-degree greedy (control)",
                "change_only_control": "Input-change/ID (control)",
                "change_random_control": "Input-change random (control)",
                "change_x_influence_ours": "Signature-degree repair (ours)",
                "hindsight_join": "Hindsight-join (oracle)",
            }.items()
        ),
        "random_control_action_matches_manuscript": sources["change_random_control"]["note"]
        == "Restrict calls to changed input signatures and choose uniformly among those cells.",
        "structural_ablation_complete": len(ablations) == 10
        and sorted({int(result["budget"]) for result in ablations}) == [32, 80]
        and len({result["condition"] for result in ablations}) == 5,
    }
    hash_checks = {}
    for filename, metadata in receipt["output_files"].items():
        path = args.result_dir / filename
        hash_checks[filename] = path.exists() and path.stat().st_size == metadata["bytes"] and sha256(path) == metadata["sha256"]
    checks["receipt_hashes_match"] = all(hash_checks.values())
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    output = {
        "status": status,
        "checks": checks,
        "hash_checks": hash_checks,
        "database_counts": counts,
        "summary_sha256": sha256(summary_path),
        "database_sha256": sha256(database_path),
        "results_sha256": sha256(results_path),
    }
    output_path = args.result_dir / "verification.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

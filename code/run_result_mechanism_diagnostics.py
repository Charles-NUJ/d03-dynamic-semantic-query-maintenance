import argparse
import csv
import hashlib
import json
from pathlib import Path

from structured_database_eval import build_orders, extract_links


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    artifact_dir = code_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-results",
        type=Path,
        default=artifact_dir / "results/pair_predicate_results.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=artifact_dir / "results/predictions.json",
    )
    parser.add_argument(
        "--revisions",
        type=Path,
        default=artifact_dir / "inputs/revisions.json",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=artifact_dir / "inputs/frozen_pairs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifact_dir / "results/result_mechanism_diagnostics",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.pair_results.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))["revisions"]
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    links_by_page = {
        int(pair["pageid"]): extract_links(
            revisions[str(pair["new_revid"])]["content"]
        )
        for pair in pairs
    }
    orders, features = build_orders(rows, predictions, revisions, links_by_page)

    method_orders = {
        "input_change_id": "change_only_control",
        "link_degree": "influence_only",
        "proxy_uncertainty": "proxy_priority",
        "proxy_degree": "proxy_x_influence",
        "signature_degree_ours": "change_x_influence_ours",
    }
    budgets = [32, 80, 145, 160]
    changed_count = sum(int(item["input_changed"]) for item in features)
    total_changed_degree = sum(
        int(item["input_changed"]) * int(item["influence"])
        for item in features
    )
    total_transition_degree = sum(
        int(row["label_flip"]) * int(features[index]["influence"])
        for index, row in enumerate(rows)
    )
    total_transitions = sum(int(row["label_flip"]) for row in rows)

    records = []
    selected_sets = {}
    for method, order_name in method_orders.items():
        for budget in budgets:
            call_cap = min(budget, changed_count) if method in {
                "input_change_id", "signature_degree_ours"
            } else budget
            selected = set(orders[order_name][:call_cap])
            selected_sets[(method, budget)] = selected
            changed_selected = sum(
                int(features[index]["input_changed"]) for index in selected
            )
            transition_selected = sum(
                int(rows[index]["label_flip"]) for index in selected
            )
            changed_degree_selected = sum(
                int(features[index]["input_changed"])
                * int(features[index]["influence"])
                for index in selected
            )
            transition_degree_selected = sum(
                int(rows[index]["label_flip"])
                * int(features[index]["influence"])
                for index in selected
            )
            records.append(
                {
                    "method": method,
                    "budget": budget,
                    "calls": len(selected),
                    "stable_selected": len(selected) - changed_selected,
                    "changed_selected": changed_selected,
                    "transition_selected": transition_selected,
                    "transition_missed": total_transitions - transition_selected,
                    "changed_degree_selected": changed_degree_selected,
                    "changed_degree_total": total_changed_degree,
                    "changed_degree_coverage": (
                        changed_degree_selected / total_changed_degree
                        if total_changed_degree > 0 else 0.0
                    ),
                    "transition_degree_selected": transition_degree_selected,
                    "transition_degree_missed": (
                        total_transition_degree - transition_degree_selected
                    ),
                    "filter_error_numerator": total_transitions - transition_selected,
                    "join_error_numerator": (
                        total_transition_degree - transition_degree_selected
                    ),
                }
            )

    explanations = {
        "link_degree_flat_32_to_80": {
            "additional_calls": len(
                selected_sets[("link_degree", 80)]
                - selected_sets[("link_degree", 32)]
            ),
            "additional_transition_cells": sum(
                int(rows[index]["label_flip"])
                for index in (
                    selected_sets[("link_degree", 80)]
                    - selected_sets[("link_degree", 32)]
                )
            ),
        },
        "filter_tie_at_32": {
            "our_transition_selected": next(
                item["transition_selected"] for item in records
                if item["method"] == "signature_degree_ours" and item["budget"] == 32
            ),
            "proxy_transition_selected": next(
                item["transition_selected"] for item in records
                if item["method"] == "proxy_uncertainty" and item["budget"] == 32
            ),
            "our_transition_degree_selected": next(
                item["transition_degree_selected"] for item in records
                if item["method"] == "signature_degree_ours" and item["budget"] == 32
            ),
            "proxy_transition_degree_selected": next(
                item["transition_degree_selected"] for item in records
                if item["method"] == "proxy_uncertainty" and item["budget"] == 32
            ),
        },
        "low_budget_gate_effect": {
            "link_degree_stable_calls": next(
                item["stable_selected"] for item in records
                if item["method"] == "link_degree" and item["budget"] == 32
            ),
            "our_stable_calls": next(
                item["stable_selected"] for item in records
                if item["method"] == "signature_degree_ours" and item["budget"] == 32
            ),
            "link_degree_join_error": next(
                item["join_error_numerator"] for item in records
                if item["method"] == "link_degree" and item["budget"] == 32
            ),
            "our_join_error": next(
                item["join_error_numerator"] for item in records
                if item["method"] == "signature_degree_ours" and item["budget"] == 32
            ),
        },
    }
    payload = {
        "status": "FINISHED",
        "purpose": "Explain recorded deterministic result differences without new semantic inference.",
        "inputs": {
            "pair_results": {"path": args.pair_results.name, "sha256": sha256(args.pair_results)},
            "predictions": {"path": args.predictions.name, "sha256": sha256(args.predictions)},
            "revisions": {"path": args.revisions.name, "sha256": sha256(args.revisions)},
            "pairs": {"path": args.pairs.name, "sha256": sha256(args.pairs)},
        },
        "totals": {
            "cells": len(rows),
            "changed_cells": changed_count,
            "transitions": total_transitions,
            "changed_degree": total_changed_degree,
            "transition_degree": total_transition_degree,
        },
        "records": records,
        "explanations": explanations,
    }
    json_path = args.output_dir / "result_mechanism_diagnostics.json"
    csv_path = args.output_dir / "result_mechanism_diagnostics.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"status": "FINISHED", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

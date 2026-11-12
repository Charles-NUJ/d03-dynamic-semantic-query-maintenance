import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from structured_database_eval import (
    build_orders,
    extract_links,
    full_output_masses,
    query_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    artifact_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-results", type=Path,
        default=artifact_dir / "results/pair_predicate_results.json",
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=artifact_dir / "results/predictions.json",
    )
    parser.add_argument(
        "--revisions", type=Path,
        default=artifact_dir / "inputs/revisions.json",
    )
    parser.add_argument(
        "--pairs", type=Path,
        default=artifact_dir / "inputs/frozen_pairs.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=artifact_dir / "results/occurrence_importance",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.pair_results.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    revision_payload = json.loads(args.revisions.read_text(encoding="utf-8"))
    revisions = revision_payload["revisions"]
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
    links_by_page = {
        int(pair["pageid"]): extract_links(
            revisions[str(pair["new_revid"])]["content"]
        )
        for pair in pairs
    }
    orders, features = build_orders(rows, predictions, revisions, links_by_page)
    full_filter_mass, full_join_mass, full_group_mass = full_output_masses(
        rows, links_by_page
    )
    if min(full_filter_mass, full_join_mass, full_group_mass) <= 0:
        raise ValueError("external result denominators must be positive")

    changed = [i for i, item in enumerate(features) if int(item["input_changed"])]
    ours_80 = orders["change_x_influence_ours"][:80]
    reuse_all = query_metrics(
        rows, [], links_by_page, full_filter_mass, full_join_mass, full_group_mass
    )
    changed_full = query_metrics(
        rows, changed, links_by_page,
        full_filter_mass, full_join_mass, full_group_mass,
    )
    ours_80_metrics = query_metrics(
        rows, ours_80, links_by_page,
        full_filter_mass, full_join_mass, full_group_mass,
    )

    by_page = defaultdict(lambda: {
        "cells": 0,
        "changed_cells": 0,
        "transition_cells": 0,
        "transition_join_rows": 0,
    })
    title_by_page = {int(pair["pageid"]): pair["title"] for pair in pairs}
    for index, row in enumerate(rows):
        pageid = int(row["pageid"])
        degree = int(features[index]["influence"])
        item = by_page[pageid]
        item["cells"] += 1
        item["changed_cells"] += int(row["input_changed"])
        item["transition_cells"] += int(row["label_flip"])
        item["transition_join_rows"] += int(row["label_flip"]) * degree

    document_records = []
    for pageid in sorted(by_page):
        document_records.append({
            "pageid": pageid,
            "title": title_by_page[pageid],
            "reference_degree": len(links_by_page[pageid]),
            **by_page[pageid],
        })

    new_times = [
        parse_timestamp(revisions[str(pair["new_revid"])]["timestamp"])
        for pair in pairs
    ]
    duration_seconds = (max(new_times) - min(new_times)).total_seconds()
    transition_degrees = sorted(
        (
            int(features[index]["influence"])
            for index, row in enumerate(rows)
            if int(row["label_flip"])
        ),
        reverse=True,
    )
    total_transition_degree = sum(transition_degrees)

    summary = {
        "status": "FINISHED",
        "purpose": (
            "Measure occurrence and reader-visible importance in the frozen natural "
            "commit without treating one trace as a production-frequency estimate."
        ),
        "inputs": {
            "pair_results": {"path": args.pair_results.name, "sha256": sha256(args.pair_results)},
            "predictions": {"path": args.predictions.name, "sha256": sha256(args.predictions)},
            "revisions": {"path": args.revisions.name, "sha256": sha256(args.revisions)},
            "pairs": {"path": args.pairs.name, "sha256": sha256(args.pairs)},
        },
        "window": {
            "documents": len(pairs),
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60.0,
        },
        "occurrence": {
            "semantic_cells": len(rows),
            "stable_cells": len(rows) - len(changed),
            "stable_fraction": (len(rows) - len(changed)) / len(rows),
            "changed_cells": len(changed),
            "changed_fraction": len(changed) / len(rows),
            "transition_cells": sum(int(row["label_flip"]) for row in rows),
            "transition_fraction_all_cells": (
                sum(int(row["label_flip"]) for row in rows) / len(rows)
            ),
            "transition_fraction_changed_cells": (
                sum(int(row["label_flip"]) for row in rows) / len(changed)
            ),
            "documents_with_changed_inputs": sum(
                item["changed_cells"] > 0 for item in document_records
            ),
            "documents_with_transitions": sum(
                item["transition_cells"] > 0 for item in document_records
            ),
            "transitions_on_unchanged_inputs": sum(
                int(row["label_flip"]) * (1 - int(row["input_changed"]))
                for row in rows
            ),
        },
        "external_importance": {
            "full_filter_rows": full_filter_mass,
            "full_join_rows": full_join_mass,
            "full_group_mass": full_group_mass,
            "reuse_all": reuse_all,
            "changed_full_calls": len(changed),
            "changed_full": changed_full,
            "ours_80_calls": len(ours_80),
            "ours_80": ours_80_metrics,
            "call_reduction_changed_full_vs_all": 1.0 - len(changed) / len(rows),
            "call_reduction_ours_80_vs_all": 1.0 - len(ours_80) / len(rows),
            "call_reduction_ours_80_vs_changed_full": 1.0 - len(ours_80) / len(changed),
            "top_1_transition_degree_fraction": (
                sum(transition_degrees[:1]) / total_transition_degree
            ),
            "top_5_transition_degree_fraction": (
                sum(transition_degrees[:5]) / total_transition_degree
            ),
            "top_10_transition_degree_fraction": (
                sum(transition_degrees[:10]) / total_transition_degree
            ),
        },
        "boundary": (
            "These values establish occurrence and external query impact in one frozen "
            "natural window. They do not estimate production frequency, SLA value, or "
            "end-to-end latency."
        ),
    }
    json_path = args.output_dir / "occurrence_importance_summary.json"
    csv_path = args.output_dir / "document_occurrence.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(document_records[0]))
        writer.writeheader()
        writer.writerows(document_records)
    print(json.dumps({
        "status": "FINISHED",
        "output": str(args.output_dir),
        "occurrence": summary["occurrence"],
        "external_importance": summary["external_importance"],
    }, indent=2))


if __name__ == "__main__":
    main()

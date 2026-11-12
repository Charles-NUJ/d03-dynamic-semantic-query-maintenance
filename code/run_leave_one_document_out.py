import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import structured_database_eval as core


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ARTIFACT / "inputs/frozen_pairs.json")
    parser.add_argument("--revisions", type=Path, default=ARTIFACT / "inputs/revisions.json")
    parser.add_argument(
        "--pair-results", type=Path, default=ARTIFACT / "results/pair_predicate_results.json"
    )
    parser.add_argument("--predictions", type=Path, default=ARTIFACT / "results/predictions.json")
    parser.add_argument(
        "--output-dir", type=Path, default=ARTIFACT / "results/document_omission"
    )
    parser.add_argument("--budget", type=int, default=80)
    args = parser.parse_args()

    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    revision_payload = json.loads(args.revisions.read_text(encoding="utf-8"))
    all_rows = json.loads(args.pair_results.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs"]
    revisions = revision_payload["revisions"]
    pair_by_page = {int(pair["pageid"]): pair for pair in pairs}
    links_by_page = {
        pageid: core.extract_links(revisions[str(pair["new_revid"])]["content"])
        for pageid, pair in pair_by_page.items()
    }

    method_orders = {
        "signature_degree_ours": "change_x_influence_ours",
        "input_change_id": "change_only_control",
        "link_degree": "influence_only",
        "proxy_degree": "proxy_x_influence",
    }
    records = []
    for omitted_pageid in sorted(pair_by_page):
        rows = [row for row in all_rows if int(row["pageid"]) != omitted_pageid]
        remaining_links = {
            pageid: links for pageid, links in links_by_page.items() if pageid != omitted_pageid
        }
        orders, features = core.build_orders(rows, predictions, revisions, remaining_links)
        filter_mass, join_mass, group_mass = core.full_output_masses(rows, remaining_links)
        if min(filter_mass, join_mass, group_mass) <= 0:
            raise ValueError(f"nonpositive output denominator after omitting page {omitted_pageid}")
        changed_count = sum(item["input_changed"] for item in features)
        budget = min(args.budget, len(rows))
        omitted_rows = [row for row in all_rows if int(row["pageid"]) == omitted_pageid]
        common = {
            "omitted_pageid": omitted_pageid,
            "omitted_title": pair_by_page[omitted_pageid]["title"],
            "omitted_degree": len(links_by_page[omitted_pageid]),
            "omitted_changed_cells": sum(int(row["input_changed"]) for row in omitted_rows),
            "omitted_transitions": sum(int(row["label_flip"]) for row in omitted_rows),
            "remaining_cells": len(rows),
            "remaining_changed_cells": changed_count,
            "budget": budget,
            "full_filter_mass": filter_mass,
            "full_join_mass": join_mass,
            "full_group_mass": group_mass,
        }
        for method, order_name in method_orders.items():
            call_cap = min(budget, changed_count) if method in {
                "signature_degree_ours", "input_change_id"
            } else budget
            selected = orders[order_name][:call_cap]
            metrics = core.query_metrics(
                rows, selected, remaining_links, filter_mass, join_mass, group_mass
            )
            records.append({
                **common,
                "method": method,
                "selected": len(selected),
                **metrics,
            })

    by_method = defaultdict(list)
    by_page = defaultdict(dict)
    for record in records:
        value = 100.0 * record["join_row_error_fraction"]
        by_method[record["method"]].append(value)
        by_page[record["omitted_pageid"]][record["method"]] = value

    controls = ["input_change_id", "link_degree", "proxy_degree"]
    pairwise = {}
    for control in controls:
        gains = [
            values[control] - values["signature_degree_ours"]
            for _, values in sorted(by_page.items())
        ]
        pairwise[control] = {
            "gain_pp": distribution(gains),
            **compare(gains),
        }
    best_gains = [
        min(values[control] for control in controls) - values["signature_degree_ours"]
        for _, values in sorted(by_page.items())
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "leave_one_document_out.csv"
    json_path = args.output_dir / "leave_one_document_out.json"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "FINISHED",
        "design": "Leave one document out at a fixed 80-call cap; reuse frozen semantic outcomes and recompute only selection and query errors.",
        "documents": len(pair_by_page),
        "rows": len(records),
        "budget": args.budget,
        "methods": list(method_orders),
        "join_error_percent": {
            method: distribution(values) for method, values in sorted(by_method.items())
        },
        "ours_against_controls": pairwise,
        "ours_against_best_control": {
            "gain_pp": distribution(best_gains),
            **compare(best_gains),
        },
        "input_sha256": {
            "pairs": sha256(args.pairs),
            "revisions": sha256(args.revisions),
            "pair_results": sha256(args.pair_results),
            "predictions": sha256(args.predictions),
        },
        "script_sha256": sha256(Path(__file__)),
        "boundary": "The 64 omissions overlap and are a sensitivity analysis, not 64 independent workload traces or evidence of production frequency.",
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

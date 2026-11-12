import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.25)
    args = parser.parse_args()

    package = args.package.resolve()
    artifact = package / "artifact" if (package / "artifact").is_dir() else package
    code_dir = artifact / "code"
    sys.path.insert(0, str(code_dir))
    import structured_database_eval as core

    inputs = artifact / "inputs"
    results = artifact / "results"
    paths = {
        "pairs": inputs / "frozen_pairs.json",
        "revisions": inputs / "revisions.json",
        "pair_results": results / "pair_predicate_results.json",
        "predictions": results / "predictions.json",
    }
    pair_payload = json.loads(paths["pairs"].read_text(encoding="utf-8"))
    revision_payload = json.loads(paths["revisions"].read_text(encoding="utf-8"))
    rows = json.loads(paths["pair_results"].read_text(encoding="utf-8"))
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    revisions = revision_payload["revisions"]
    pairs = sorted(
        pair_payload["pairs"],
        key=lambda item: (datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")), int(item["pageid"])),
    )
    if len(pairs) != 64:
        raise ValueError(f"expected 64 page pairs, got {len(pairs)}")

    pair_by_page = {int(pair["pageid"]): pair for pair in pairs}
    links_by_page = {
        pageid: core.extract_links(revisions[str(pair["new_revid"])]["content"])
        for pageid, pair in pair_by_page.items()
    }
    methods = {
        "ttl": "ttl_control",
        "edit": "edit_control",
        "proxy_uncertainty": "proxy_priority",
        "signature_degree_ours": "change_x_influence_ours",
        "input_change_id": "change_only_control",
        "link_degree": "influence_only",
        "ttl_degree": "ttl_x_influence",
        "edit_degree": "edit_x_influence",
        "proxy_degree": "proxy_x_influence",
    }
    split_pairs = {
        "early_32": pairs[:32],
        "late_32": pairs[32:],
    }
    split_pageids = {
        name: [int(pair["pageid"]) for pair in members]
        for name, members in split_pairs.items()
    }
    records = []
    for split_name, members in split_pairs.items():
        pageids = {int(pair["pageid"]) for pair in members}
        split_rows = [row for row in rows if int(row["pageid"]) in pageids]
        split_links = {pageid: links_by_page[pageid] for pageid in pageids}
        orders, features = core.build_orders(split_rows, predictions, revisions, split_links)
        filter_mass, join_mass, group_mass = core.full_output_masses(split_rows, split_links)
        if min(filter_mass, join_mass, group_mass) <= 0:
            raise ValueError(f"nonpositive denominator in {split_name}")
        changed = sum(int(item["input_changed"]) for item in features)
        transitions = sum(int(row["label_flip"]) for row in split_rows)
        budget = round(args.budget_fraction * len(split_rows))
        degrees = [len(split_links[int(pair["pageid"])]) for pair in members]
        common = {
            "split": split_name,
            "start_time": members[0]["timestamp"],
            "end_time": members[-1]["timestamp"],
            "documents": len(members),
            "cells": len(split_rows),
            "stable_cells": len(split_rows) - changed,
            "changed_cells": changed,
            "transition_cells": transitions,
            "budget": budget,
            "budget_fraction_of_cells": budget / len(split_rows),
            "full_filter_mass": filter_mass,
            "full_join_mass": join_mass,
            "degree_mean": statistics.fmean(degrees),
            "degree_cv": statistics.pstdev(degrees) / statistics.fmean(degrees),
        }
        for method, order_name in methods.items():
            call_cap = min(budget, changed) if method in {
                "signature_degree_ours",
                "input_change_id",
            } else budget
            selected = orders[order_name][:call_cap]
            metrics = core.query_metrics(
                split_rows,
                selected,
                split_links,
                filter_mass,
                join_mass,
                group_mass,
            )
            changed_degree_total = sum(
                feature["input_changed"] * feature["influence"] for feature in features
            )
            changed_degree_selected = sum(
                features[index]["input_changed"] * features[index]["influence"]
                for index in selected
            )
            transition_degree_selected = sum(
                int(split_rows[index]["label_flip"]) * features[index]["influence"]
                for index in selected
            )
            records.append({
                **common,
                "method": method,
                "selected": len(selected),
                "stable_selected": sum(
                    1 - features[index]["input_changed"] for index in selected
                ),
                "transition_selected": sum(
                    int(split_rows[index]["label_flip"]) for index in selected
                ),
                "changed_degree_coverage": (
                    changed_degree_selected / changed_degree_total
                    if changed_degree_total > 0 else 1.0
                ),
                "transition_degree_selected": transition_degree_selected,
                **metrics,
            })

    by_split = {}
    for split_name in split_pairs:
        group = {row["method"]: row for row in records if row["split"] == split_name}
        control = min(
            group[name]["join_row_error_fraction"]
            for name in ("input_change_id", "link_degree", "proxy_degree")
        )
        ours = group["signature_degree_ours"]["join_row_error_fraction"]
        by_split[split_name] = {
            "structure": {
                key: group["signature_degree_ours"][key]
                for key in (
                    "documents",
                    "cells",
                    "stable_cells",
                    "changed_cells",
                    "transition_cells",
                    "budget",
                    "full_join_mass",
                    "degree_mean",
                    "degree_cv",
                )
            },
            "join_error_percent": {
                name: 100.0 * row["join_row_error_fraction"]
                for name, row in group.items()
            },
            "mechanism": {
                name: {
                    "stable_selected": row["stable_selected"],
                    "transition_selected": row["transition_selected"],
                    "changed_degree_coverage": row["changed_degree_coverage"],
                    "transition_degree_selected": row["transition_degree_selected"],
                }
                for name, row in group.items()
            },
            "ours_gain_over_best_control_pp": 100.0 * (control - ours),
        }

    payload = {
        "status": "FINISHED",
        "design": (
            "Sort the 64 frozen page pairs by commit timestamp, split them into two disjoint "
            "32-document halves, and recompute nine causal policies at 25% of full calls per half."
        ),
        "scope": (
            "Internal sensitivity within one 12.1-minute trace. The halves are disjoint in documents "
            "but share the same collection process, model, predicates, and time window; they are not "
            "independent production traces."
        ),
        "splits": by_split,
        "split_pageids": split_pageids,
        "records": records,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "script_sha256_before_output": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

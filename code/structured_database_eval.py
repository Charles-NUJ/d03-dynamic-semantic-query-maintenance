import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


WIKILINK = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
EXCLUDED_NAMESPACES = {
    "category", "file", "image", "template", "help", "portal", "special",
    "talk", "user", "wikipedia", "mediawiki", "module", "draft", "book",
    "timedtext", "topic",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_target(raw: str) -> str | None:
    target = raw.split("|", 1)[0].strip()
    target = target.split("#", 1)[0].strip().lstrip(":")
    target = " ".join(target.replace("_", " ").split())
    if not target:
        return None
    if ":" in target:
        prefix = target.split(":", 1)[0].strip().lower()
        if prefix in EXCLUDED_NAMESPACES:
            return None
    return target[0].upper() + target[1:]


def extract_links(wikitext: str) -> list[str]:
    links = set()
    for match in WIKILINK.finditer(wikitext):
        target = normalize_target(match.group(1))
        if target:
            links.add(target)
    return sorted(links)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def group_l1_error(rows, selected, links_by_page) -> int:
    signed = Counter()
    selected_set = set(selected)
    for index, row in enumerate(rows):
        if index in selected_set:
            continue
        delta = int(row["new_label"]) - int(row["old_label"])
        if delta == 0:
            continue
        predicate = int(row["predicate_index"])
        for target in links_by_page[int(row["pageid"])]:
            signed[(predicate, target)] += delta
    return int(sum(abs(value) for value in signed.values()))


def query_metrics(rows, selected, links_by_page, full_filter_mass, full_join_mass, full_group_mass):
    selected_set = set(selected)
    filter_error = 0
    join_error = 0
    for index, row in enumerate(rows):
        if index in selected_set:
            continue
        if int(row["new_label"]) == int(row["old_label"]):
            continue
        filter_error += 1
        join_error += len(links_by_page[int(row["pageid"])])
    group_error = group_l1_error(rows, selected, links_by_page)
    return {
        "filter_error": int(filter_error),
        "filter_error_fraction": float(filter_error / max(1, full_filter_mass)),
        "join_row_error": int(join_error),
        "join_row_error_fraction": float(join_error / max(1, full_join_mass)),
        "group_l1_error": int(group_error),
        "group_l1_error_fraction": float(group_error / max(1, full_group_mass)),
    }


def make_database(path: Path, pairs, rows, predictions, revisions, links_by_page):
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        CREATE TABLE document (
            pageid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            old_revid INTEGER NOT NULL,
            new_revid INTEGER NOT NULL,
            old_timestamp TEXT NOT NULL,
            new_timestamp TEXT NOT NULL,
            old_content_sha256 TEXT NOT NULL,
            new_content_sha256 TEXT NOT NULL
        );
        CREATE TABLE semantic_cell_old (
            pageid INTEGER NOT NULL,
            predicate_index INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            old_label INTEGER NOT NULL CHECK(old_label IN (0,1)),
            old_yes_probability REAL NOT NULL,
            source_revid INTEGER NOT NULL,
            PRIMARY KEY(pageid, predicate_index)
        );
        CREATE TABLE semantic_input_state (
            pageid INTEGER NOT NULL,
            predicate_index INTEGER NOT NULL,
            old_input_sha256 TEXT NOT NULL,
            new_input_sha256 TEXT NOT NULL,
            input_changed INTEGER NOT NULL CHECK(input_changed IN (0,1)),
            PRIMARY KEY(pageid, predicate_index)
        );
        CREATE TABLE oracle_semantic_cell_new (
            pageid INTEGER NOT NULL,
            predicate_index INTEGER NOT NULL,
            new_label INTEGER NOT NULL CHECK(new_label IN (0,1)),
            source_revid INTEGER NOT NULL,
            PRIMARY KEY(pageid, predicate_index)
        );
        CREATE TABLE reference_new (
            pageid INTEGER NOT NULL,
            target TEXT NOT NULL,
            PRIMARY KEY(pageid, target)
        );
        """
    )
    prediction_by_key = {
        (int(item["revid"]), int(item["predicate_index"])): item
        for item in predictions
    }
    pair_by_page = {int(item["pageid"]): item for item in pairs}
    for pageid, pair in pair_by_page.items():
        old_revision = revisions[str(pair["old_revid"])]
        new_revision = revisions[str(pair["new_revid"])]
        connection.execute(
            "INSERT INTO document VALUES (?,?,?,?,?,?,?,?)",
            (
                pageid,
                pair["title"],
                int(pair["old_revid"]),
                int(pair["new_revid"]),
                old_revision["timestamp"],
                new_revision["timestamp"],
                hashlib.sha256(old_revision["content"].encode("utf-8")).hexdigest(),
                hashlib.sha256(new_revision["content"].encode("utf-8")).hexdigest(),
            ),
        )
        connection.executemany(
            "INSERT INTO reference_new VALUES (?,?)",
            [(pageid, target) for target in links_by_page[pageid]],
        )
    for row in rows:
        pageid = int(row["pageid"])
        predicate_index = int(row["predicate_index"])
        old_prediction = prediction_by_key[(int(row["old_revid"]), predicate_index)]
        new_prediction = prediction_by_key[(int(row["new_revid"]), predicate_index)]
        connection.execute(
            "INSERT INTO semantic_cell_old VALUES (?,?,?,?,?,?)",
            (
                pageid,
                predicate_index,
                row["predicate"],
                int(row["old_label"]),
                float(old_prediction["yes_probability"]),
                int(row["old_revid"]),
            ),
        )
        connection.execute(
            "INSERT INTO semantic_input_state VALUES (?,?,?,?,?)",
            (
                pageid,
                predicate_index,
                old_prediction["visible_text_sha256"],
                new_prediction["visible_text_sha256"],
                int(row["input_changed"]),
            ),
        )
        connection.execute(
            "INSERT INTO oracle_semantic_cell_new VALUES (?,?,?,?)",
            (pageid, predicate_index, int(row["new_label"]), int(row["new_revid"])),
        )
    connection.commit()
    connection.close()


def build_orders(rows, predictions, revisions, links_by_page):
    prediction_by_key = {
        (int(item["revid"]), int(item["predicate_index"])): item
        for item in predictions
    }
    selection_time = max(
        parse_timestamp(revisions[str(row["new_revid"])]["timestamp"])
        for row in rows
    )
    features = []
    for index, row in enumerate(rows):
        old_prediction = prediction_by_key[(int(row["old_revid"]), int(row["predicate_index"]))]
        old_time = parse_timestamp(revisions[str(row["old_revid"])]["timestamp"])
        relative_edit = abs(int(row["newlen"]) - int(row["oldlen"])) / max(1, int(row["oldlen"]))
        degree = len(links_by_page[int(row["pageid"])])
        features.append(
            {
                "index": index,
                "ttl": old_time.timestamp(),
                "ttl_age": max(0.0, (selection_time - old_time).total_seconds()),
                "input_changed": int(row["input_changed"]),
                "edit": relative_edit,
                "old_confidence_priority": 0.5 - abs(float(old_prediction["yes_probability"]) - 0.5),
                "influence": degree,
                "hindsight_filter": int(row["label_flip"]),
                "hindsight_join": int(row["label_flip"]) * degree,
            }
        )

    def descending(name):
        return sorted(range(len(rows)), key=lambda i: (-features[i][name], i))

    return {
        "ttl_control": sorted(range(len(rows)), key=lambda i: (features[i]["ttl"], i)),
        "edit_control": descending("edit"),
        "proxy_priority": descending("old_confidence_priority"),
        "influence_only": descending("influence"),
        "change_only_control": descending("input_changed"),
        "ttl_x_influence": sorted(
            range(len(rows)),
            key=lambda i: (-(features[i]["ttl_age"] * features[i]["influence"]), i),
        ),
        "edit_x_influence": sorted(
            range(len(rows)),
            key=lambda i: (-(features[i]["edit"] * features[i]["influence"]), i),
        ),
        "proxy_x_influence": sorted(
            range(len(rows)),
            key=lambda i: (-(features[i]["old_confidence_priority"] * features[i]["influence"]), i),
        ),
        "change_x_influence_ours": sorted(
            range(len(rows)),
            key=lambda i: (
                -features[i]["input_changed"],
                -features[i]["influence"],
                i,
            ),
        ),
        "hindsight_filter": descending("hindsight_filter"),
        "hindsight_join": descending("hindsight_join"),
    }, features


def full_output_masses(rows, links_by_page):
    full_filter_mass = sum(int(row["new_label"]) for row in rows)
    full_join_mass = sum(
        int(row["new_label"]) * len(links_by_page[int(row["pageid"])])
        for row in rows
    )
    group_counts = Counter()
    for row in rows:
        if not int(row["new_label"]):
            continue
        predicate = int(row["predicate_index"])
        for target in links_by_page[int(row["pageid"])]:
            group_counts[(predicate, target)] += 1
    full_group_mass = sum(abs(value) for value in group_counts.values())
    return full_filter_mass, full_join_mass, full_group_mass


def main():
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent
    default_pairs = artifact_dir / "inputs/frozen_pairs.json"
    default_revisions = artifact_dir / "inputs/revisions.json"
    default_pair_results = artifact_dir / "results/pair_predicate_results.json"
    default_predictions = artifact_dir / "results/predictions.json"
    default_output = artifact_dir / "reproduced_results/main"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=default_pairs)
    parser.add_argument("--revisions", type=Path, default=default_revisions)
    parser.add_argument("--pair-results", type=Path, default=default_pair_results)
    parser.add_argument("--predictions", type=Path, default=default_predictions)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--random-repetitions", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    revision_payload = json.loads(args.revisions.read_text(encoding="utf-8"))
    rows = json.loads(args.pair_results.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs"]
    revisions = revision_payload["revisions"]

    links_by_page = {}
    for pair in pairs:
        pageid = int(pair["pageid"])
        new_content = revisions[str(pair["new_revid"])]["content"]
        links_by_page[pageid] = extract_links(new_content)

    database_path = args.output_dir / "structured_database.sqlite"
    make_database(database_path, pairs, rows, predictions, revisions, links_by_page)
    orders, features = build_orders(rows, predictions, revisions, links_by_page)
    full_filter_mass, full_join_mass, full_group_mass = full_output_masses(rows, links_by_page)
    if full_filter_mass <= 0 or full_join_mass <= 0 or full_group_mass <= 0:
        raise ValueError("all external metric denominators must be positive")

    method_sources = {
        "reuse_all": {
            "display": "Reuse-all (endpoint)",
            "kind": "complete endpoint",
            "sources": [],
            "note": "Keep every stored semantic value after the document update.",
        },
        "lineage_only": {
            "display": "Lineage-full refresh [Metaxy, mlidea]",
            "kind": "ability-matched literature adaptation",
            "sources": ["metaxy", "mlidea"],
            "note": "Recompute every semantic cell reached by exact version lineage; this is not a reproduction of either system.",
        },
        "uniform_clean": {
            "display": "SVC-style uniform [SVC]",
            "kind": "ability-matched literature adaptation",
            "sources": ["svc"],
            "note": "Transfer only uniform sample selection from SVC; omit its aggregate estimator and outlier index, and score the maintained filter and join directly.",
        },
        "ttl_control": {
            "display": "TTL greedy (control)",
            "kind": "system control",
            "sources": [],
            "note": "Prioritize the oldest stored source version.",
        },
        "edit_control": {
            "display": "Edit greedy (control)",
            "kind": "problem-specific control",
            "sources": [],
            "note": "Prioritize relative document-length change without reading the new semantic value.",
        },
        "proxy_priority": {
            "display": "Proxy uncertainty [SUPG]",
            "kind": "ability-matched literature adaptation",
            "sources": ["supg"],
            "note": "Transfer proxy-guided evaluation from SUPG and rank stored prediction uncertainty; this is not a reproduction of SUPG.",
        },
        "influence_only": {
            "display": "Link-degree greedy (control)",
            "kind": "structural ablation",
            "sources": [],
            "note": "Rank only by the number of current main-namespace links in the materialized join relation.",
        },
        "change_only_control": {
            "display": "Input-change/ID (control)",
            "kind": "structural ablation",
            "sources": [],
            "note": "Prioritize cells whose fixed operator input signature changed; break ties by cell identifier.",
        },
        "change_random_control": {
            "display": "Input-change random (control)",
            "kind": "randomized structural control",
            "sources": [],
            "note": "Restrict calls to changed input signatures and choose uniformly among those cells.",
        },
        "ttl_x_influence": {
            "display": "TTL-degree greedy (control)",
            "kind": "query-aware control",
            "sources": [],
            "note": "Rank by stored-value age times current link degree.",
        },
        "edit_x_influence": {
            "display": "Edit-degree greedy (control)",
            "kind": "query-aware control",
            "sources": [],
            "note": "Rank by relative document-length change times current link degree.",
        },
        "proxy_x_influence": {
            "display": "Proxy-degree [SUPG]",
            "kind": "ability-matched query-aware adaptation",
            "sources": ["supg"],
            "note": "Multiply stored prediction uncertainty by current link degree; no external system is reproduced.",
        },
        "change_x_influence_ours": {
            "display": "Signature-degree repair (ours)",
            "kind": "our method",
            "sources": [],
            "note": "Use exact fixed-input invariance, then minimize the worst-case residual join-row error by link degree.",
        },
        "hindsight_filter": {
            "display": "Hindsight-filter (oracle)",
            "kind": "noncausal lower bound",
            "sources": [],
            "note": "Read the hidden transition before selection and prioritize every changed label.",
        },
        "hindsight_join": {
            "display": "Hindsight-join (oracle)",
            "kind": "noncausal lower bound",
            "sources": [],
            "note": "Read the hidden transition and its current join degree before selection.",
        },
    }

    budgets = [0, 32, 80, 160, 240, 320]
    changed_indices = [i for i, item in enumerate(features) if item["input_changed"]]
    changed_count = len(changed_indices)
    gated_methods = {"change_only_control", "change_x_influence_ours"}
    results = []
    for budget in budgets:
        deterministic = {
            "reuse_all": [],
            "lineage_only": list(range(len(rows))) if budget == len(rows) else [],
        }
        for name, order in orders.items():
            call_cap = min(budget, changed_count) if name in gated_methods else budget
            deterministic[name] = order[:call_cap]
        for method, selected in deterministic.items():
            if method == "lineage_only" and budget != len(rows):
                continue
            metrics = query_metrics(
                rows, selected, links_by_page,
                full_filter_mass, full_join_mass, full_group_mass,
            )
            results.append(
                {
                    "method": method,
                    "budget": budget,
                    "budget_fraction": budget / len(rows),
                    "selected": len(selected),
                    **metrics,
                    "repetitions": 1,
                }
            )
        if budget not in (0, len(rows)):
            samples = defaultdict(list)
            change_samples = defaultdict(list)
            for repeat in range(args.random_repetitions):
                rng = random.Random(20261101 + repeat)
                selected = rng.sample(range(len(rows)), budget)
                metrics = query_metrics(
                    rows, selected, links_by_page,
                    full_filter_mass, full_join_mass, full_group_mass,
                )
                for key, value in metrics.items():
                    samples[key].append(value)
                change_selected = rng.sample(
                    changed_indices, min(budget, len(changed_indices))
                )
                change_metrics = query_metrics(
                    rows, change_selected, links_by_page,
                    full_filter_mass, full_join_mass, full_group_mass,
                )
                for key, value in change_metrics.items():
                    change_samples[key].append(value)
            result = {
                "method": "uniform_clean",
                "budget": budget,
                "budget_fraction": budget / len(rows),
                "selected": budget,
                "repetitions": args.random_repetitions,
            }
            for key, values in samples.items():
                result[key] = float(sum(values) / len(values))
                mean = result[key]
                result[key + "_std"] = float(
                    math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1))
                )
            results.append(result)
            change_result = {
                "method": "change_random_control",
                "budget": budget,
                "budget_fraction": budget / len(rows),
                "selected": min(budget, len(changed_indices)),
                "repetitions": args.random_repetitions,
            }
            for key, values in change_samples.items():
                change_result[key] = float(sum(values) / len(values))
                mean = change_result[key]
                change_result[key + "_std"] = float(
                    math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1))
                )
            results.append(change_result)

    structural_ablation = []
    ablation_repetitions = 200
    for budget in [32, 80]:
        actual_orders = {
            "input_change": orders["change_only_control"][:budget],
            "link_degree": orders["influence_only"][:budget],
            "input_change_then_degree": orders["change_x_influence_ours"][:budget],
        }
        for name, selected in actual_orders.items():
            structural_ablation.append(
                {
                    "condition": name,
                    "budget": budget,
                    "repetitions": 1,
                    **query_metrics(
                        rows, selected, links_by_page,
                        full_filter_mass, full_join_mass, full_group_mass,
                    ),
                }
            )
        for condition in ["input_change_then_shuffled_degree", "shuffled_input_change_then_degree"]:
            sampled = defaultdict(list)
            for repeat in range(ablation_repetitions):
                rng = random.Random(20261201 + repeat)
                shuffled = list(range(len(rows)))
                rng.shuffle(shuffled)
                if condition == "input_change_then_shuffled_degree":
                    order = sorted(
                        range(len(rows)),
                        key=lambda i: (
                            -features[i]["input_changed"],
                            -features[shuffled[i]]["influence"],
                            i,
                        ),
                    )
                else:
                    order = sorted(
                        range(len(rows)),
                        key=lambda i: (
                            -features[shuffled[i]]["input_changed"],
                            -features[i]["influence"],
                            i,
                        ),
                    )
                metrics = query_metrics(
                    rows, order[:budget], links_by_page,
                    full_filter_mass, full_join_mass, full_group_mass,
                )
                for key, value in metrics.items():
                    sampled[key].append(value)
            item = {
                "condition": condition,
                "budget": budget,
                "repetitions": ablation_repetitions,
            }
            for key, values in sampled.items():
                item[key] = float(sum(values) / len(values))
            structural_ablation.append(item)

    degree_values = [len(links_by_page[int(pair["pageid"])]) for pair in pairs]
    mean_degree = sum(degree_values) / len(degree_values)
    changed_input_cells = sum(item["input_changed"] for item in features)
    transitions_on_unchanged_inputs = sum(
        int(row["label_flip"]) * (1 - int(row["input_changed"]))
        for row in rows
    )
    structure = {
        "pages": len(pairs),
        "semantic_cells": len(rows),
        "reference_rows": sum(degree_values),
        "unique_targets": len({target for links in links_by_page.values() for target in links}),
        "degree_min": min(degree_values),
        "degree_median": sorted(degree_values)[len(degree_values) // 2],
        "degree_max": max(degree_values),
        "degree_mean": mean_degree,
        "degree_cv": math.sqrt(sum((x - mean_degree) ** 2 for x in degree_values) / len(degree_values)) / max(mean_degree, 1e-12),
        "changed_input_cells": changed_input_cells,
        "unchanged_input_cells": len(rows) - changed_input_cells,
        "input_change_fraction": changed_input_cells / len(rows),
        "transitions_on_unchanged_inputs": transitions_on_unchanged_inputs,
        "semantic_transitions": sum(int(row["label_flip"]) for row in rows),
        "full_filter_mass": full_filter_mass,
        "full_join_mass": full_join_mass,
        "full_group_mass": full_group_mass,
    }

    source_map_path = args.output_dir / "method_source_map.json"
    source_map_path.write_text(json.dumps(method_sources, indent=2) + "\n", encoding="utf-8")
    reference_path = args.output_dir / "reference_relation.json"
    reference_payload = [
        {"pageid": pageid, "target": target}
        for pageid in sorted(links_by_page)
        for target in links_by_page[pageid]
    ]
    reference_path.write_text(json.dumps(reference_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "method_results.csv"
    fieldnames = sorted({key for result in results for key in result})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    ablation_path = args.output_dir / "structural_ablation.csv"
    ablation_fields = sorted({key for result in structural_ablation for key in result})
    with ablation_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ablation_fields)
        writer.writeheader()
        writer.writerows(structural_ablation)

    summary = {
        "status": "FINISHED",
        "experiment_id": "D03-STRUCTURED-DATABASE-V2-20261102",
        "information_order": {
            "observable_before_action": [
                "old and new document versions", "fixed-operator input signatures",
                "old semantic value and confidence",
                "query lineage", "current extracted main-namespace link relation",
            ],
            "hidden_before_action": [
                "new semantic value", "semantic transition", "full-recompute query result",
            ],
            "control": "subset of affected semantic cells recomputed before publication",
            "resource_contract": "the call budget is a cap over the affected set; exact-reuse methods stop after every changed input has been called",
        },
        "query": {
            "filter": "semantic cells with maintained value one",
            "join": "filter output joined with distinct current main-namespace links by pageid",
            "aggregate": "count joined rows by predicate and target",
        },
        "inputs": {
            "pairs": {"file": args.pairs.name, "sha256": sha256(args.pairs)},
            "revisions": {"file": args.revisions.name, "sha256": sha256(args.revisions)},
            "pair_results": {"file": args.pair_results.name, "sha256": sha256(args.pair_results)},
            "predictions": {"file": args.predictions.name, "sha256": sha256(args.predictions)},
        },
        "structure": structure,
        "methods": method_sources,
        "results": results,
        "structural_ablation": structural_ablation,
        "boundaries": [
            "The Reference relation is extracted from the current revision wikitext and is traceable, but it is one workload rather than a production query log.",
            "Literature-labelled baselines are ability-matched adaptations; no external system is claimed to be reproduced.",
            "Input-signature invariance assumes a deterministic semantic operator with fixed model, prompt, schema, and decoding rule.",
            "SHA-256 input identity is treated as collision-free within the frozen catalog.",
            "The full-recompute labels are fixed-operator outputs rather than human semantic truth.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    receipt = {
        "status": "FINISHED",
        "output_files": {},
    }
    for path in [database_path, source_map_path, reference_path, csv_path, ablation_path, summary_path]:
        receipt["output_files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    receipt_path = args.output_dir / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINISHED", "structure": structure, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

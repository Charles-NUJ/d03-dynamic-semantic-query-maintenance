import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path


METHODS = {
    "random_all": "SVC-style uniform: gate off, degree off",
    "random_changed": "Changed-first random: gate on, degree off",
    "degree_all": "Link-degree: gate off, degree on",
    "change_degree": "Signature-degree repair (ours): gate on, degree on",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_eval_module(path: Path):
    spec = importlib.util.spec_from_file_location("structured_database_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values):
    return float(sum(values) / len(values))


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)))


def summarize(samples, repetitions):
    item = {"repetitions": repetitions}
    for key, values in samples.items():
        item[key] = mean(values)
        item[key + "_std"] = sample_std(values)
        item[key + "_ci95"] = 1.96 * item[key + "_std"] / math.sqrt(repetitions)
    return item


def write_csv(path: Path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_artifact = script_dir.parent if script_dir.name == "code" else None
    parser.add_argument("--artifact-dir", type=Path, default=default_artifact)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.artifact_dir is None:
        raise ValueError("--artifact-dir is required outside an artifact/code directory")
    if args.repetitions <= 1:
        raise ValueError("repetitions must exceed one")

    artifact = args.artifact_dir.resolve()
    code_path = artifact / "code/structured_database_eval.py"
    pairs_path = artifact / "inputs/frozen_pairs.json"
    revisions_path = artifact / "inputs/revisions.json"
    pair_results_path = artifact / "results/pair_predicate_results.json"
    predictions_path = artifact / "results/predictions.json"
    method_results_path = artifact / "results/method_results.csv"
    module = load_eval_module(code_path)

    pair_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    revision_payload = json.loads(revisions_path.read_text(encoding="utf-8"))
    rows = json.loads(pair_results_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs"]
    revisions = revision_payload["revisions"]

    links_by_page = {}
    for pair in pairs:
        pageid = int(pair["pageid"])
        content = revisions[str(pair["new_revid"])]["content"]
        links_by_page[pageid] = module.extract_links(content)

    orders, features = module.build_orders(rows, predictions, revisions, links_by_page)
    full_filter, full_join, full_group = module.full_output_masses(rows, links_by_page)
    if min(full_filter, full_join, full_group) <= 0:
        raise ValueError("external denominators must be positive")

    changed = [i for i, item in enumerate(features) if item["input_changed"]]
    stable = [i for i, item in enumerate(features) if not item["input_changed"]]
    total_changed_degree = sum(features[i]["influence"] for i in changed)
    if not changed or not stable or total_changed_degree <= 0:
        raise ValueError("the factorial design requires changed, stable, and positive-degree cells")

    budgets = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 145, 160]
    results = []

    def diagnostics(selected):
        chosen = set(selected)
        selected_changed = [i for i in chosen if features[i]["input_changed"]]
        selected_stable = [i for i in chosen if not features[i]["input_changed"]]
        changed_degree = sum(features[i]["influence"] for i in selected_changed)
        selected_transition_count = sum(int(rows[i]["label_flip"]) for i in chosen)
        selected_transition_degree = sum(
            int(rows[i]["label_flip"]) * features[i]["influence"] for i in chosen
        )
        budget = len(chosen)
        return {
            "stable_calls": len(selected_stable),
            "stable_call_fraction": len(selected_stable) / max(1, budget),
            "changed_calls": len(selected_changed),
            "changed_degree_selected": changed_degree,
            "changed_degree_coverage": changed_degree / total_changed_degree,
            "worst_case_residual_rows": total_changed_degree - changed_degree,
            "selected_transition_count_hindsight": selected_transition_count,
            "selected_transition_degree_hindsight": selected_transition_degree,
        }

    def evaluate(selected):
        values = module.query_metrics(
            rows, selected, links_by_page, full_filter, full_join, full_group
        )
        values.update(diagnostics(selected))
        return values

    for budget in budgets:
        deterministic = {
            "degree_all": orders["influence_only"][:budget],
            "change_degree": orders["change_x_influence_ours"][:min(budget, len(changed))],
        }
        for method, selected in deterministic.items():
            results.append(
                {
                    "method": method,
                    "display": METHODS[method],
                    "gate": int(method == "change_degree"),
                    "degree": 1,
                    "budget": budget,
                    "budget_fraction": budget / len(rows),
                    "selected": len(selected),
                    **evaluate(selected),
                    "repetitions": 1,
                }
            )

        sampled = {"random_all": defaultdict(list), "random_changed": defaultdict(list)}
        for repeat in range(args.repetitions):
            # Reuse the frozen experiment seed schedule at every budget. This
            # preserves the published B=32 and B=80 random controls exactly.
            rng = random.Random(20261101 + repeat)
            random_all = rng.sample(range(len(rows)), budget)
            random_changed = rng.sample(changed, min(budget, len(changed)))
            for method, selected in (
                ("random_all", random_all),
                ("random_changed", random_changed),
            ):
                for key, value in evaluate(selected).items():
                    sampled[method][key].append(value)
        for method in ("random_all", "random_changed"):
            results.append(
                {
                    "method": method,
                    "display": METHODS[method],
                    "gate": int(method == "random_changed"),
                    "degree": 0,
                    "budget": budget,
                    "budget_fraction": budget / len(rows),
                    "selected": min(budget, len(changed)) if method == "random_changed" else budget,
                    **summarize(sampled[method], args.repetitions),
                }
            )

    results.sort(key=lambda item: (item["budget"], item["gate"], item["degree"]))
    by_budget = defaultdict(dict)
    for item in results:
        by_budget[item["budget"]][item["method"]] = item

    decomposition = []
    for budget in budgets:
        cells = by_budget[budget]
        e00 = 100.0 * cells["random_all"]["join_row_error_fraction"]
        e10 = 100.0 * cells["random_changed"]["join_row_error_fraction"]
        e01 = 100.0 * cells["degree_all"]["join_row_error_fraction"]
        e11 = 100.0 * cells["change_degree"]["join_row_error_fraction"]
        gate_random = e00 - e10
        degree_all = e00 - e01
        gate_degree = e01 - e11
        degree_changed = e10 - e11
        combined = e00 - e11
        overlap = gate_random + degree_all - combined
        decomposition.append(
            {
                "budget": budget,
                "uniform_error_pct": e00,
                "gate_only_error_pct": e10,
                "degree_only_error_pct": e01,
                "both_error_pct": e11,
                "gate_gain_without_degree_pp": gate_random,
                "degree_gain_without_gate_pp": degree_all,
                "gate_gain_with_degree_pp": gate_degree,
                "degree_gain_with_gate_pp": degree_changed,
                "combined_gain_pp": combined,
                "overlap_pp": overlap,
                "uniform_stable_calls": cells["random_all"]["stable_calls"],
                "degree_only_stable_calls": cells["degree_all"]["stable_calls"],
                "both_stable_calls": cells["change_degree"]["stable_calls"],
                "gate_only_calls": cells["random_changed"]["selected"],
                "both_calls": cells["change_degree"]["selected"],
                "gate_only_unused_budget": budget - cells["random_changed"]["selected"],
                "both_unused_budget": budget - cells["change_degree"]["selected"],
                "gate_only_changed_degree_coverage": cells["random_changed"]["changed_degree_coverage"],
                "degree_only_changed_degree_coverage": cells["degree_all"]["changed_degree_coverage"],
                "both_changed_degree_coverage": cells["change_degree"]["changed_degree_coverage"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.output_dir / "mechanism_attribution.csv"
    decomposition_csv = args.output_dir / "factorial_decomposition.csv"
    write_csv(result_csv, results)
    write_csv(decomposition_csv, decomposition)

    existing = list(csv.DictReader(method_results_path.open(encoding="utf-8")))
    existing_lookup = {
        (row["method"], int(row["budget"])): float(row["join_row_error_fraction"])
        for row in existing
    }
    mapping = {
        "random_all": "uniform_clean",
        "random_changed": "change_random_control",
        "degree_all": "influence_only",
        "change_degree": "change_x_influence_ours",
    }
    key_checks = {}
    for budget in (32, 80):
        for method, old_name in mapping.items():
            observed = by_budget[budget][method]["join_row_error_fraction"]
            expected = existing_lookup[(old_name, budget)]
            key_checks[f"matches_existing_{method}_b{budget}"] = abs(observed - expected) < 0.004

    checks = {
        **key_checks,
        "dense_budget_grid_present": len(budgets) == 12,
        "gate_eliminates_stable_calls_below_threshold": all(
            by_budget[budget][method]["stable_calls"] == 0
            for budget in budgets if budget <= len(changed)
            for method in ("random_changed", "change_degree")
        ),
        "our_coverage_is_maximal": all(
            by_budget[budget]["change_degree"]["changed_degree_coverage"] + 1e-12
            >= by_budget[budget][method]["changed_degree_coverage"]
            for budget in budgets
            for method in METHODS
        ),
        "our_error_is_minimal_among_four": all(
            by_budget[budget]["change_degree"]["join_row_error_fraction"] <=
            by_budget[budget][method]["join_row_error_fraction"] + 1e-12
            for budget in budgets
            for method in METHODS
        ),
        "changed_threshold_zero_error": (
            by_budget[len(changed)]["random_changed"]["join_row_error"] == 0
            and by_budget[len(changed)]["change_degree"]["join_row_error"] == 0
        ),
        "selection_respects_budget_cap": all(item["selected"] <= item["budget"] for item in results),
        "gated_methods_stop_at_changed_set": all(
            by_budget[budget][method]["selected"] == min(budget, len(changed))
            for budget in budgets
            for method in ("random_changed", "change_degree")
        ),
        "positive_denominators": min(full_filter, full_join, full_group) > 0,
    }
    status = "VERIFIED" if all(checks.values()) else "FAILED"
    record = {
        "status": status,
        "design": "2x2 stable-base gate by degree ordering under a shared call budget",
        "methods": METHODS,
        "budgets": budgets,
        "random_repetitions": args.repetitions,
        "counts": {
            "affected_cells": len(rows),
            "changed_cells": len(changed),
            "stable_cells": len(stable),
            "total_changed_degree": total_changed_degree,
            "full_filter_mass": full_filter,
            "full_join_mass": full_join,
            "full_group_mass": full_group,
        },
        "checks": checks,
        "selected_budget_rows": {
            str(budget): next(item for item in decomposition if item["budget"] == budget)
            for budget in (32, 80, 145)
        },
        "hashes": {
            "structured_database_eval": sha256(code_path),
            "pairs": sha256(pairs_path),
            "revisions": sha256(revisions_path),
            "pair_results": sha256(pair_results_path),
            "predictions": sha256(predictions_path),
            "method_results": sha256(method_results_path),
            "mechanism_attribution_csv": sha256(result_csv),
            "factorial_decomposition_csv": sha256(decomposition_csv),
        },
        "boundary": (
            "The factor experiment explains allocation inside one frozen workload. "
            "Hindsight transition columns diagnose realized outcomes and never enter selection."
        ),
    }
    output_json = args.output_dir / "mechanism_attribution.json"
    output_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

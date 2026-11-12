from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("d03_structured_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_window(evaluator, paths: dict[str, Path]):
    pair_payload = json.loads(paths["pairs"].read_text(encoding="utf-8"))
    revision_payload = json.loads(paths["revisions"].read_text(encoding="utf-8"))
    rows = json.loads(paths["rows"].read_text(encoding="utf-8"))
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    pairs = pair_payload["pairs"]
    revisions = revision_payload["revisions"]
    links_by_page = {}
    for pair in pairs:
        pageid = int(pair["pageid"])
        content = revisions[str(pair["new_revid"])]["content"]
        links_by_page[pageid] = evaluator.extract_links(content)
    orders, features = evaluator.build_orders(rows, predictions, revisions, links_by_page)
    masses = evaluator.full_output_masses(rows, links_by_page)
    if min(masses) <= 0:
        raise ValueError("Every external metric denominator must be positive.")
    return rows, links_by_page, orders, features, masses


def greedy_selection(features, budget: int) -> tuple[int, ...]:
    changed = [i for i, item in enumerate(features) if int(item["input_changed"]) == 1]
    k = min(budget, len(changed))
    return tuple(sorted(changed, key=lambda i: (-int(features[i]["influence"]), i))[:k])


def exact_dynamic_program(features, budget: int) -> tuple[int, ...]:
    changed = [i for i, item in enumerate(features) if int(item["input_changed"]) == 1]
    k = min(budget, len(changed))
    unavailable = None
    states: list[tuple[int, tuple[int, ...]] | None] = [unavailable] * (k + 1)
    states[0] = (0, ())
    processed = 0
    for index in changed:
        processed += 1
        weight = int(features[index]["influence"])
        for used in range(min(k, processed), 0, -1):
            previous = states[used - 1]
            if previous is None:
                continue
            candidate = (previous[0] + weight, previous[1] + (index,))
            current = states[used]
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                states[used] = candidate
    if states[k] is None:
        raise RuntimeError("Exact dynamic program did not reach the requested budget.")
    return states[k][1]


def dual_price_iteration(features, budget: int) -> tuple[tuple[int, ...], int, int]:
    changed = [i for i, item in enumerate(features) if int(item["input_changed"]) == 1]
    k = min(budget, len(changed))
    if k == 0:
        return (), 0, 0
    weights = {i: int(features[i]["influence"]) for i in changed}
    low = min(weights.values())
    high = max(weights.values())
    iterations = 0
    while low < high:
        iterations += 1
        midpoint = (low + high + 1) // 2
        if sum(value >= midpoint for value in weights.values()) >= k:
            low = midpoint
        else:
            high = midpoint - 1
    price = low
    strict = sorted(i for i, value in weights.items() if value > price)
    ties = sorted(i for i, value in weights.items() if value == price)
    selected = tuple(strict + ties[: k - len(strict)])
    if len(selected) != k:
        raise RuntimeError("Dual price did not recover the requested cardinality.")
    return selected, price, iterations


def selection_sha(selected: tuple[int, ...]) -> str:
    payload = ",".join(str(value) for value in sorted(selected)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def objective(features, selected: tuple[int, ...]) -> int:
    return int(
        sum(
            int(features[i]["input_changed"]) * int(features[i]["influence"])
            for i in selected
        )
    )


def benchmark(selector, repeats: int):
    timings = []
    last = None
    metadata = {}
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = selector()
        timings.append(time.perf_counter_ns() - start)
        if isinstance(result, tuple) and len(result) == 3 and isinstance(result[0], tuple):
            last = result[0]
            metadata = {"dual_price": result[1], "iterations": result[2]}
        else:
            last = result
    ordered = sorted(timings)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return last, {
        "median_selection_us": statistics.median(timings) / 1000.0,
        "p95_selection_us": ordered[p95_index] / 1000.0,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=300)
    args = parser.parse_args()
    root = args.package_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator_path = root / "artifact/code/structured_database_eval.py"
    evaluator = load_module(evaluator_path)
    windows = {
        "window_1": {
            "pairs": root / "artifact/inputs/frozen_pairs.json",
            "revisions": root / "artifact/inputs/revisions.json",
            "rows": root / "artifact/results/pair_predicate_results.json",
            "predictions": root / "artifact/results/predictions.json",
        },
        "window_2": {
            "pairs": root / "artifact/inputs/second_window_20261110/frozen/frozen_pairs.json",
            "revisions": root / "artifact/inputs/second_window_20261110/revision_content/revisions.json",
            "rows": root / "artifact/results/second_window_20261110/inference/pair_predicate_results.json",
            "predictions": root / "artifact/results/second_window_20261110/inference/predictions.json",
        },
    }
    records = []
    window_records = {}
    for window_name, paths in windows.items():
        rows, links, orders, features, masses = load_window(evaluator, paths)
        changed = [i for i, item in enumerate(features) if int(item["input_changed"]) == 1]
        window_records[window_name] = {
            "cells": len(rows),
            "changed_cells": len(changed),
            "positive_degree_changed_cells": sum(int(features[i]["influence"]) > 0 for i in changed),
            "full_filter_mass": masses[0],
            "full_join_mass": masses[1],
            "full_group_mass": masses[2],
            "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        }
        for budget in (32, 80):
            selectors = {
                "signature_degree_greedy_ours": lambda f=features, b=budget: greedy_selection(f, b),
                "central_exact_dp_ours": lambda f=features, b=budget: exact_dynamic_program(f, b),
                "dual_price_iteration_ours": lambda f=features, b=budget: dual_price_iteration(f, b),
            }
            greedy_reference = tuple(orders["change_x_influence_ours"][: min(budget, len(changed))])
            for method, selector in selectors.items():
                selected, timing = benchmark(selector, args.repeats)
                if method == "signature_degree_greedy_ours" and selected != greedy_reference:
                    raise AssertionError("Reimplemented greedy rule differs from the registered order.")
                metrics = evaluator.query_metrics(rows, selected, links, *masses)
                total_weight = sum(int(features[i]["influence"]) for i in changed)
                record = {
                    "window": window_name,
                    "method": method,
                    "budget": budget,
                    "selected": len(selected),
                    "selected_weight": objective(features, selected),
                    "residual_weight": total_weight - objective(features, selected),
                    "selection_sha256": selection_sha(selected),
                    **metrics,
                    **timing,
                }
                records.append(record)

    fields = sorted({key for row in records for key in row})
    csv_path = output / "same_contract_strong_methods.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    record_path = output / "execution_record.json"
    record = {
        "status": "FINISHED",
        "contract": "Same frozen inputs, observable signatures and degrees, call budgets, and external filter/join/group errors.",
        "methods": {
            "signature_degree_greedy_ours": "Our registered one-pass degree ordering inside the changed-input layer.",
            "central_exact_dp_ours": "Our independent centralized dynamic program for the same unit-cost robust objective.",
            "dual_price_iteration_ours": "Our integer dual-price threshold iteration for the same objective; index order resolves ties.",
        },
        "budgets": [32, 80],
        "timing_repeats": args.repeats,
        "windows": window_records,
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
        "result": {"path": str(csv_path), "sha256": sha256(csv_path), "rows": len(records)},
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FINISHED", "rows": len(records), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()

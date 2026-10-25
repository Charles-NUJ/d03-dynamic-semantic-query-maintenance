"""D03 revision O: heterogeneous exact-state-quotient stress experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
D = R / "raw_results/major_o_v1"
V = R / "derived/major_o_v1"
A = R / "reports/major_o_current"


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rec(path):
    path = Path(path)
    return {"path": path.relative_to(R).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def prepare():
    assert not D.exists(), "major_o_v1 already exists"
    D.mkdir(parents=True)
    V.mkdir(parents=True, exist_ok=True)
    A.mkdir(parents=True, exist_ok=True)
    protocol = {
        "time": now(),
        "status": "FROZEN_BEFORE_O_EXPERIMENT",
        "purpose": "Measure exact quotient degradation as heterogeneous group count grows, without changing the controller or its safety contract.",
        "benchmark": {
            "query": "filter",
            "rows": 16384,
            "group_counts": [1, 4, 16, 64, 128, 180],
            "repetitions": 7,
            "construction": "All rows unresolved and dirty predicted zeros. Groups have distinct sizes 1..K; the last group receives the remaining rows. Each group induces separate audit and refresh keys.",
            "comparison": ["WCR", "WQCR"],
            "expected": "WCR uses 4N branch tests. WQCR uses 4K branch tests. Both select the same action.",
            "boundary": "Controlled in-memory action-selection stress test; not an end-to-end SQL, production, or adversarial worst-case benchmark.",
        },
        "seed": 2026102204,
        "deterministic": True,
        "formal_submission": False,
    }
    save(D / "protocol.json", protocol)
    inputs = [
        R / "src/d03_materialized_j.py",
        R / "raw_results/major_n_v1/summary.json",
        R / "derived/major_n_v1/VERIFICATION.json",
    ]
    save(D / "input_manifest.json", [rec(x) for x in inputs])
    snap = D / "source_snapshot"
    snap.mkdir()
    sources = []
    for name in ["run_d03_major_o.py", "verify_d03_major_o.py", "d03_materialized_j.py"]:
        src = R / "src" / name
        dst = snap / name
        shutil.copyfile(src, dst)
        sources.append({**rec(dst), "origin": f"src/{name}", "origin_sha256": sha(src)})
    save(D / "source_manifest.json", sources)
    save(D / "environment.json", {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "remote": False,
        "GPU_calls": 0,
    })
    save(D / "prepared.json", {
        "time": now(),
        "protocol_sha256": sha(D / "protocol.json"),
        "input_manifest_sha256": sha(D / "input_manifest.json"),
        "source_manifest_sha256": sha(D / "source_manifest.json"),
        "frozen_before_execution": True,
        "launch": ["workspace-python src/run_d03_major_o.py run", "workspace-python src/verify_d03_major_o.py"],
    })
    print("PREPARED_MAJOR_O", len(inputs), "inputs")


def validate_frozen():
    prepared = load(D / "prepared.json")
    assert sha(D / "protocol.json") == prepared["protocol_sha256"]
    assert sha(D / "input_manifest.json") == prepared["input_manifest_sha256"]
    assert sha(D / "source_manifest.json") == prepared["source_manifest_sha256"]
    for item in load(D / "input_manifest.json") + load(D / "source_manifest.json"):
        assert sha(R / item["path"]) == item["sha256"], item["path"]


def build_state(n, groups):
    from d03_materialized_j import State

    if groups == 1:
        sizes = [n]
    else:
        sizes = list(range(1, groups + 1))
        sizes[-1] += n - sum(sizes)
    assert len(set(sizes)) == len(sizes) and sum(sizes) == n and min(sizes) > 0
    state = object.__new__(State)
    state.query = "filter"
    state.ctx = "fixed"
    state.rule = "positive"
    state.original = {}
    state.e = {}
    rid = 0
    for gi, size in enumerate(sizes):
        group = f"g{gi:03d}"
        for _ in range(size):
            row = f"r{rid:06d}"
            state.e[row] = (f"k{rid:06d}", group, 0, -1, 1)
            rid += 1
    state.published = set()
    state.rebuild()
    return state, sizes


def run():
    from d03_materialized_j import choose

    validate_frozen()
    assert not (D / "execution_receipt.json").exists()
    protocol = load(D / "protocol.json")["benchmark"]
    n = protocol["rows"]
    raw = []
    start_all = time.perf_counter()
    for groups in protocol["group_counts"]:
        state, sizes = build_state(n, groups)
        for _ in range(2):
            choose(state, "WCR", 2 * n, 4 * n, True, True)
            choose(state, "WQCR", 2 * n, 4 * n, True, True)
        for repetition in range(protocol["repetitions"]):
            pair = {}
            for method in ["WCR", "WQCR"]:
                start = time.perf_counter_ns()
                action, tests, representatives = choose(state, method, 2 * n, 4 * n, True, True)
                elapsed = time.perf_counter_ns() - start
                pair[method] = action
                raw.append({
                    "rows": n,
                    "groups": groups,
                    "distinct_group_sizes": len(set(sizes)),
                    "repetition": repetition,
                    "method": method,
                    "action": list(action),
                    "branch_tests": tests,
                    "representatives": representatives,
                    "elapsed_ns": elapsed,
                })
            assert pair["WCR"] == pair["WQCR"]
    save(D / "heterogeneous_quotient_raw.json", raw)
    summary_rows = []
    for groups in protocol["group_counts"]:
        wcr = [x for x in raw if x["groups"] == groups and x["method"] == "WCR"]
        wqcr = [x for x in raw if x["groups"] == groups and x["method"] == "WQCR"]
        wcr_ms = float(np.median([x["elapsed_ns"] for x in wcr]) / 1e6)
        wqcr_ms = float(np.median([x["elapsed_ns"] for x in wqcr]) / 1e6)
        summary_rows.append({
            "rows": n,
            "groups": groups,
            "exact_keys": 2 * groups,
            "WCR_branch_tests": wcr[0]["branch_tests"],
            "WQCR_branch_tests": wqcr[0]["branch_tests"],
            "exact_test_reduction": wcr[0]["branch_tests"] / wqcr[0]["branch_tests"],
            "WCR_median_ms": wcr_ms,
            "WQCR_median_ms": wqcr_ms,
            "median_speedup": wcr_ms / wqcr_ms,
            "same_action_all_repetitions": all(a["action"] == b["action"] for a, b in zip(wcr, wqcr)),
        })
    summary = {
        "time": now(),
        "status": "O_EXECUTED_VERIFICATION_PENDING",
        "records": len(raw),
        "benchmark": summary_rows,
        "wall_seconds": time.perf_counter() - start_all,
        "new_GPU_calls": 0,
        "new_model_calls": 0,
        "new_labels": 0,
        "new_trajectories": 0,
        "formal_submission": False,
        "limits": protocol["boundary"],
    }
    save(D / "summary.json", summary)
    save(D / "execution_receipt.json", {**summary, "status": "O_EXECUTION_FINISHED_SEPARATE_VERIFICATION_PENDING"})
    print(json.dumps(summary, indent=2))


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["prepare", "run"])
args = parser.parse_args()
{"prepare": prepare, "run": run}[args.action]()

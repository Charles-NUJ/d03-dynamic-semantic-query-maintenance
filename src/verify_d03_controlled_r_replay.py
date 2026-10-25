"""Independent verifier for the frozen D03 Revision-R comparison.

The verifier does not import the R controller.  It checks frozen inputs and
source hashes, SQL archives, hard budgets and publication certificates, the
WCR/WQCR quotient trace, and every same-state 2x2 planner record.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import collections, hashlib, json, math, sqlite3

import numpy as np

R = Path(__file__).resolve().parents[1]
D = R / "raw_results/controlled_r_v2"
V = R / "derived/controlled_r_v2"
A = R / "reports/controlled_r_current"
METHODS = ["CF", "SC", "CR", "WCR", "WQCR", "FRANK", "SQP", "LRANK", "LQP", "PGAIN", "COST"]
W = np.array([1.0, 1.0, 0.25, 0.1, 0.25])
TAU = 0.01
TOL = 1e-8


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_simplex(v):
    """Independent Euclidean simplex projection."""
    u = np.sort(np.asarray(v, dtype=float))[::-1]
    cssv = np.cumsum(u) - 1.0
    valid = u - cssv / np.arange(1, len(u) + 1) > 0
    rho = np.flatnonzero(valid)[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(np.asarray(v, dtype=float) - theta, 0.0)


def independent_qp(G, target, cost):
    """Projected gradient reconstruction written only in this verifier."""
    n = G.shape[1]
    x = np.full(n, 1.0 / n)
    L = float(np.sum(W[:, None] * G * G) + TAU)
    assert L >= TAU > 0
    for iteration in range(1, 1201):
        grad = G.T @ (W * (G @ x - target)) + TAU * x + cost
        nxt = project_simplex(x - grad / L)
        residual = float(L * np.linalg.norm(x - nxt))
        x = nxt
        if residual <= TOL:
            break
    grad = G.T @ (W * (G @ x - target)) + TAU * x + cost
    residual = float(L * np.linalg.norm(x - project_simplex(x - grad / L)))
    gap = float(grad @ x - np.min(grad))
    return x, iteration, residual, gap


cfg = load(D / "protocol.json")
prepared = load(D / "prepared.json")
assert cfg["status"] == "FROZEN_BEFORE_R_CONTROLLED_EXECUTION"
assert cfg["methods"] == METHODS
assert sha(D / "protocol.json") == prepared["protocol_sha256"]
assert sha(D / "input_manifest.json") == prepared["input_manifest_sha256"]
assert sha(D / "source_manifest.json") == prepared["source_manifest_sha256"]
for entry in load(D / "input_manifest.json") + load(D / "source_manifest.json"):
    assert sha(R / entry["path"]) == entry["sha256"], entry["path"]

paths = sorted((D / "trajectories").glob("*.json"))
assert len(paths) == 396
counts = collections.Counter()
maximum = collections.defaultdict(float)
rows = []
traces = {}
learning_errors = collections.defaultdict(list)

for path in paths:
    trajectory = load(path)
    method = trajectory["method"]
    assert method in METHODS
    assert len(trajectory["events"]) == 9
    base = (trajectory["stream"], trajectory["query"], trajectory["fee_cap"], trajectory["write_cap"])
    trace = []
    previous_fee = previous_writes = 0.0
    for event_index, event in enumerate(trajectory["events"]):
        assert event["fee"] <= event["fee_cap"] + 1e-8
        assert event["writes"] <= event["write_cap"]
        assert len(event["steps"]) <= 2 * event["initial"]["rows"]
        if event["published"]:
            c = event["final_cert"]
            assert c[0] >= 0 and c[1] >= 0 and c[3] == 0
            counts["published"] += 1
        else:
            counts["refused"] += 1
        if event["initial"]["joint_protected"]:
            assert event["published"]
            counts["joint_protected_events"] += 1
        previous_fee = previous_writes = 0.0
        step_trace = []
        for step_index, step in enumerate(event["steps"]):
            assert step["fee_total"] + 1e-12 >= previous_fee
            assert step["write_total"] >= previous_writes
            assert step["fee_total"] <= event["fee_cap"] + 1e-8
            assert step["write_total"] <= event["write_cap"]
            previous_fee, previous_writes = step["fee_total"], step["write_total"]
            step_trace.append((step["rid"], step["kind"], step["bit"], step["fee_total"], step["write_total"]))
            feedback = step["paid_feedback"]
            if method in {"LRANK", "LQP", "PGAIN"}:
                assert feedback is not None
                assert feedback["samples_before"] >= 0
                assert 0.0 <= feedback["probability_before"] <= 1.0
                assert feedback["target_flip"] in (0, 1)
                assert math.isclose(feedback["squared_error"], (feedback["probability_before"] - feedback["target_flip"]) ** 2, abs_tol=1e-12)
                assert feedback["before_sha256"] != feedback["after_sha256"]
                learning_errors[method].append(feedback["squared_error"])
                counts["paid_learning_updates"] += 1
            else:
                assert feedback is None

            cells = step["plan"].get("controlled_cells")
            if cells is not None:
                assert method == "LQP"
                Gf = np.asarray(cells["fixed_gain_matrix"], dtype=float)
                Gl = np.asarray(cells["learned_gain_matrix"], dtype=float)
                target = np.asarray(cells["target"], dtype=float)
                cost = np.asarray(cells["cost"], dtype=float)
                assert Gf.shape == Gl.shape == (5, cells["classes"])
                assert target.shape == (5,) and cost.shape == (cells["classes"],)
                fixed_scores = cost - Gf.T @ (W * target)
                learned_scores = cost - Gl.T @ (W * target)
                xf, fit, fres, fgap = independent_qp(Gf, target, cost)
                xl, lit, lres, lgap = independent_qp(Gl, target, cost)
                fr, fq, lr, lq = (cells["fixed_rank"], cells["fixed_qp"],
                                  cells["learned_rank"], cells["learned_qp"])
                # JSON round trips can reverse np.argmin/argmax within an exact
                # symmetry class.  Check optimal value plus the controller's
                # stable stored index instead of demanding the verifier's tie.
                assert fixed_scores[fr] <= np.min(fixed_scores) + 1e-12
                assert learned_scores[lr] <= np.min(learned_scores) + 1e-12
                assert xf[fq] >= np.max(xf) - 1e-12
                assert xl[lq] >= np.max(xl) - 1e-12
                assert math.isclose(cells["fixed_rank_score"], fixed_scores[fr], abs_tol=1e-12)
                assert math.isclose(cells["learned_rank_score"], learned_scores[lr], abs_tol=1e-12)
                assert np.allclose(xf, cells["fixed_qp_mass"], atol=2e-10, rtol=0)
                assert np.allclose(xl, cells["learned_qp_mass"], atol=2e-10, rtol=0)
                assert lq == step["plan"]["selected_class"]
                assert (step["rid"], step["kind"]) == tuple(step["plan"]["candidate_actions"][lq])
                maximum["fixed_qp_residual"] = max(maximum["fixed_qp_residual"], fres)
                maximum["learned_qp_residual"] = max(maximum["learned_qp_residual"], lres)
                maximum["fixed_simplex_gap"] = max(maximum["fixed_simplex_gap"], fgap)
                maximum["learned_simplex_gap"] = max(maximum["learned_simplex_gap"], lgap)
                counts["factorial_states_checked"] += 1
                counts["fixed_rank_vs_qp_disagree"] += fr != fq
                counts["learned_rank_vs_qp_disagree"] += lr != lq
                counts["fixed_vs_learned_rank_disagree"] += fr != lr
                counts["fixed_vs_learned_qp_disagree"] += fq != lq
                counts["all_four_same"] += len({fr, fq, lr, lq}) == 1
                counts["fixed_qp_iterations_recomputed"] += fit
                counts["learned_qp_iterations_recomputed"] += lit
        trace.append((event["published"], event["refusal_reason"], event["fee"], event["writes"], tuple(step_trace)))
        counts["events"] += 1
    db = R / trajectory["database"]
    assert db.exists() and sha(db) == trajectory["database_sha256"]
    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
    connection.close()
    counts["sqlite_archives_reopened"] += 1
    traces[base + (method,)] = trace
    row = {
        "id": trajectory["id"], "stream": trajectory["stream"], "query": trajectory["query"],
        "fee_cap": trajectory["fee_cap"], "write_cap": trajectory["write_cap"], "method": method,
        "nonempty": sum(e["nonempty"] for e in trajectory["events"]),
        "served": sum(e["nonempty"] and e["published"] for e in trajectory["events"]),
        "fee": sum(e["fee"] for e in trajectory["events"]),
        "writes": sum(e["writes"] for e in trajectory["events"]),
        "steps": sum(len(e["steps"]) for e in trajectory["events"]),
        "branch_tests": sum(e["branch_tests"] for e in trajectory["events"]),
    }
    rows.append(row)

assert counts["events"] == 3564
assert counts["sqlite_archives_reopened"] == 396
assert counts["factorial_states_checked"] > 0
for key in sorted({k[:-1] for k in traces if k[-1] == "WCR"}):
    assert traces[key + ("WCR",)] == traces[key + ("WQCR",)]
    counts["WCR_WQCR_exact_trace_pairs"] += 1
assert counts["WCR_WQCR_exact_trace_pairs"] == 36

by_cell = collections.defaultdict(list)
by_method = collections.defaultdict(list)
for row in rows:
    by_cell[(row["query"], row["fee_cap"], row["write_cap"], row["method"])].append(row)
    by_method[row["method"]].append(row)

aggregate = []
for key, group in sorted(by_cell.items()):
    assert len(group) == 3 and sum(x["nonempty"] for x in group) == 24
    aggregate.append({
        "query": key[0], "fee_cap": key[1], "write_cap": key[2], "method": key[3],
        "streams": 3, "nonempty_events": 24,
        "served": sum(x["served"] for x in group),
        "fee": sum(x["fee"] for x in group), "writes": sum(x["writes"] for x in group),
        "steps": sum(x["steps"] for x in group), "branch_tests": sum(x["branch_tests"] for x in group),
    })

method_totals = []
for method in METHODS:
    group = by_method[method]
    assert len(group) == 36
    served = sum(x["served"] for x in group)
    method_totals.append({
        "method": method, "trajectories": 36, "nonempty_events": sum(x["nonempty"] for x in group),
        "served": served, "fee": sum(x["fee"] for x in group), "writes": sum(x["writes"] for x in group),
        "steps": sum(x["steps"] for x in group), "branch_tests": sum(x["branch_tests"] for x in group),
        "fee_per_served": (sum(x["fee"] for x in group) / served) if served > 0 else None,
        "writes_per_served": (sum(x["writes"] for x in group) / served) if served > 0 else None,
    })

cell_lookup = {(x["query"], x["fee_cap"], x["write_cap"], x["method"]): x for x in aggregate}
comparisons = []
for left, right, factor in [
    ("LQP", "SQP", "paid ridge under QP, endogenous histories"),
    ("LQP", "LRANK", "QP under paid ridge, endogenous histories"),
    ("SQP", "FRANK", "QP under fixed p=0.5, endogenous histories"),
    ("LRANK", "FRANK", "paid ridge under linear rank, endogenous histories"),
    ("LQP", "PGAIN", "full QP versus D03 proxy-gain specialization"),
    ("LQP", "COST", "full QP versus D03 cost-order specialization"),
    ("LQP", "WQCR", "adaptive QP versus exact quotient baseline"),
]:
    differences = []
    for query in cfg["queries"]:
        for fee_cap in cfg["fee_caps"]:
            for write_cap in cfg["write_caps"][query]:
                a = cell_lookup[(query, fee_cap, write_cap, left)]
                b = cell_lookup[(query, fee_cap, write_cap, right)]
                differences.append({"query": query, "fee_cap": fee_cap, "write_cap": write_cap,
                                    "served_delta": a["served"] - b["served"], "fee_delta": a["fee"] - b["fee"],
                                    "writes_delta": a["writes"] - b["writes"], "steps_delta": a["steps"] - b["steps"]})
    comparisons.append({
        "left": left, "right": right, "interpretation": factor, "paired_cells": 12,
        "served_delta_total": sum(x["served_delta"] for x in differences),
        "fee_delta_total": sum(x["fee_delta"] for x in differences),
        "writes_delta_total": sum(x["writes_delta"] for x in differences),
        "steps_delta_total": sum(x["steps_delta"] for x in differences),
        "cell_differences": differences,
    })

save(V / "aggregate.json", aggregate)
save(V / "trajectory_metrics.json", rows)
save(V / "method_totals.json", method_totals)
save(V / "paired_comparisons.json", comparisons)

verification = {
    "time": datetime.now().astimezone().isoformat(),
    "status": "PASS_R_CONTROLLED_FACTORIAL_LITERATURE_SPECIALIZATIONS_AND_SQL_SAFETY",
    "counts": dict(counts), "maxima": dict(maximum),
    "learning": {m: {"paid_samples": len(v), "prequential_MSE": sum(v) / len(v)} for m, v in learning_errors.items()},
    "raw_receipt_sha256": sha(D / "execution_receipt.json"),
    "protocol_sha256": sha(D / "protocol.json"),
    "scope": "Frozen 396-trajectory experiment; 396 read-only SQLite checks; hard budgets and final publication certificates; 36 exact WCR/WQCR trace pairs; every recorded same-state 2x2 cell independently recomputed. Literature-inspired PGAIN and COST are D03 specializations; no external engine or code was executed. End-to-end factor differences are descriptive because policy histories diverge.",
    "AI_REVIEW": True, "human_reviewed": False, "formal_submission": False,
}
save(V / "VERIFICATION.json", verification)
save(A / "EXECUTION_VERIFICATION.json", verification)
print(json.dumps(verification, ensure_ascii=False, indent=2))

"""D03 R: controlled planners and literature-inspired D03 specializations.

All methods share the exact eligibility gates.  FRANK completes the fixed/learned
by linear/QP control matrix.  PGAIN and COST are D03-specific algorithms, not
reproductions of an external engine.
"""
from __future__ import annotations

import numpy as np

from d03_materialized_j import State, SQL, digest, choose as baseline_choose
from d03_adaptive_k import (
    Learner, WEIGHTS, eligible, objective, planning, solve,
)

BASELINES = {"CF", "SC", "CR", "WCR", "WQCR"}
LEARNED = {"LRANK", "LQP", "PGAIN"}
QP_METHODS = {"SQP", "LQP"}


def _linear_choice(G, target, cost):
    score = cost - G.T @ (WEIGHTS * target)
    return int(np.argmin(score)), score


def _qp_choice(G, target, cost):
    x, info = solve(G, target, cost)
    return int(np.argmax(x)), x, info


def controlled_cells(s, candidates, learner, rf, rw, bf, bw):
    """Evaluate all four planner cells on exactly the same state and learner."""
    pf = [0.5] * len(candidates)
    pl = [learner.predict(s, c["action"]) for c in candidates]
    Gf, target, cost = planning(s, candidates, pf, rf, rw, bf, bw)
    Gl, target_l, cost_l = planning(s, candidates, pl, rf, rw, bf, bw)
    assert np.array_equal(target, target_l) and np.array_equal(cost, cost_l)
    fr, fs = _linear_choice(Gf, target, cost)
    fq, fx, fi = _qp_choice(Gf, target, cost)
    lr, ls = _linear_choice(Gl, target, cost)
    lq, lx, li = _qp_choice(Gl, target, cost)
    return {
        "classes": len(candidates),
        "fixed_rank": fr,
        "fixed_qp": fq,
        "learned_rank": lr,
        "learned_qp": lq,
        "fixed_rank_score": float(fs[fr]),
        "learned_rank_score": float(ls[lr]),
        "fixed_qp_iterations": fi["iterations"],
        "learned_qp_iterations": li["iterations"],
        "fixed_qp_converged": fi["converged"],
        "learned_qp_converged": li["converged"],
        "fixed_qp_mass": fx.tolist(),
        "learned_qp_mass": lx.tolist(),
        "fixed_probabilities": pf,
        "learned_probabilities": pl,
        # These arrays let the separate verifier reconstruct each cell without
        # importing this controller or replaying an endogenous learner history.
        "fixed_gain_matrix": Gf.tolist(),
        "learned_gain_matrix": Gl.tolist(),
        "target": target.tolist(),
        "cost": cost.tolist(),
    }


def choose(s, method, rf, rw, fp, jp, bf, bw, learner):
    if method in BASELINES:
        a, tests, reps = baseline_choose(s, method, rf, rw, fp, jp)
        return a, {"mode": method, "branch_tests": tests, "representatives": reps}

    candidates, tests = eligible(s, rf, rw, fp, jp)
    if not candidates:
        return None, {"mode": "NO_ADMISSIBLE_ACTION", "branch_tests": tests, "classes": 0}
    terminals = [c for c in candidates if c["terminal"]]
    if terminals:
        c = min(terminals, key=lambda z: (z["fee"], z["commit"], z["order"]))
        return c["action"], {"mode": "TERMINAL", "branch_tests": tests, "classes": len(candidates)}

    if method == "COST":
        j = min(range(len(candidates)), key=lambda k: (
            candidates[k]["fee"], candidates[k]["commit"], candidates[k]["order"]
        ))
        return candidates[j]["action"], {
            "mode": "COST_ORDER", "branch_tests": tests, "classes": len(candidates),
            "selected_class": j,
        }

    probabilities = [learner.predict(s, c["action"]) for c in candidates] if method in LEARNED else [0.5] * len(candidates)
    G, target, cost = planning(s, candidates, probabilities, rf, rw, bf, bw)
    if method in {"FRANK", "LRANK"}:
        j, score = _linear_choice(G, target, cost)
        info = {"mode": "LINEAR_RANK", "scores": score.tolist()}
    elif method == "PGAIN":
        values = []
        for k in range(len(candidates)):
            e = np.zeros(len(candidates)); e[k] = 1.0
            values.append(objective(G, target, cost, e))
        j = int(np.argmin(values)); info = {"mode": "PROXY_GAIN", "one_step_objectives": values}
    else:
        j, mass, info = _qp_choice(G, target, cost)
        info.update(mode="QP", allocation=mass.tolist())
    info.update(
        branch_tests=tests, classes=len(candidates),
        candidate_actions=[list(c["action"]) for c in candidates],
        probabilities=probabilities, selected_class=j,
    )
    if method == "LQP":
        info["controlled_cells"] = controlled_cells(s, candidates, learner, rf, rw, bf, bw)
        assert info["controlled_cells"]["learned_qp"] == j
    return candidates[j]["action"], info

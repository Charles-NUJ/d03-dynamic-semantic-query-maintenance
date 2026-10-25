"""Independent record-level verification for D03 revision O experiment."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

R = Path(__file__).resolve().parents[1]
D = R / "raw_results/major_o_v1"
V = R / "derived/major_o_v1"
A = R / "reports/major_o_current"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


prepared = load(D / "prepared.json")
assert sha(D / "protocol.json") == prepared["protocol_sha256"]
assert sha(D / "input_manifest.json") == prepared["input_manifest_sha256"]
assert sha(D / "source_manifest.json") == prepared["source_manifest_sha256"]
for item in load(D / "input_manifest.json") + load(D / "source_manifest.json"):
    assert sha(R / item["path"]) == item["sha256"], item["path"]

protocol = load(D / "protocol.json")["benchmark"]
raw = load(D / "heterogeneous_quotient_raw.json")
summary = load(D / "summary.json")
expected_records = len(protocol["group_counts"]) * protocol["repetitions"] * 2
assert len(raw) == summary["records"] == expected_records
checks = []
for groups in protocol["group_counts"]:
    rows = [x for x in raw if x["groups"] == groups]
    assert len(rows) == protocol["repetitions"] * 2
    for repetition in range(protocol["repetitions"]):
        wcr = next(x for x in rows if x["repetition"] == repetition and x["method"] == "WCR")
        wqcr = next(x for x in rows if x["repetition"] == repetition and x["method"] == "WQCR")
        assert wcr["action"] == wqcr["action"]
        assert wcr["branch_tests"] == 4 * protocol["rows"]
        assert wqcr["branch_tests"] == 4 * groups
        assert wcr["representatives"] == 2 * protocol["rows"]
        assert wqcr["representatives"] == 2 * groups
        assert wcr["elapsed_ns"] > 0 and wqcr["elapsed_ns"] > 0
    checks.append({
        "groups": groups,
        "exact_keys": 2 * groups,
        "WCR_branch_tests": 4 * protocol["rows"],
        "WQCR_branch_tests": 4 * groups,
        "same_action_all_repetitions": True,
    })

reported = {x["groups"]: x for x in summary["benchmark"]}
for check in checks:
    row = reported[check["groups"]]
    assert row["WCR_branch_tests"] == check["WCR_branch_tests"]
    assert row["WQCR_branch_tests"] == check["WQCR_branch_tests"]
    assert row["exact_keys"] == check["exact_keys"]
    assert row["same_action_all_repetitions"]

out = {
    "time": datetime.now().astimezone().isoformat(),
    "status": "PASS_O_HETEROGENEOUS_EXACT_QUOTIENT_STRESS",
    "records_checked": len(raw),
    "group_settings_checked": len(checks),
    "checks": checks,
    "input_files_reopened": len(load(D / "input_manifest.json")),
    "source_files_reopened": len(load(D / "source_manifest.json")),
    "raw_files": {
        name: {"path": f"raw_results/major_o_v1/{name}", "sha256": sha(D / name), "bytes": (D / name).stat().st_size}
        for name in ["heterogeneous_quotient_raw.json", "summary.json"]
    },
    "AI_REVIEW": True,
    "human_reviewed": False,
    "formal_submission": False,
    "limits": "Independent manifest and record verification of the controlled action-selection benchmark. It does not establish end-to-end database speed, production behavior, global novelty, or adversarial worst-case reduction.",
}
save(V / "VERIFICATION.json", out)
save(A / "EXECUTION_VERIFICATION.json", out)
print(json.dumps(out, indent=2))

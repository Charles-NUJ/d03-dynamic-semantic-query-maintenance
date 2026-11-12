from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((run / "records.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    checks = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    check("experiment finished", summary["status"] == "EXECUTED_AWAITING_INDEPENDENT_VERIFICATION", summary["status"])
    check("eighteen method runs", len(records) == 18, len(records))
    methods = {item["method"] for item in records}
    expected = {"reuse_all", "input_change_id", "link_degree", "proxy_degree", "signature_degree_ours", "full_recompute"}
    check("method set exact", methods == expected, sorted(methods))
    check("three repetitions each", all(sum(x["method"] == method for x in records) == 3 for method in expected), {method: sum(x["method"] == method for x in records) for method in expected})
    check("positive external denominators", all(x["join_mass"] > 0 and x["filter_mass"] > 0 for x in records), sorted({(x["join_mass"], x["filter_mass"]) for x in records}))
    check("ours uses eighty changed calls", all(x["calls"] == 80 and x["stable_calls"] == 0 for x in records if x["method"] == "signature_degree_ours"), [(x["calls"], x["stable_calls"]) for x in records if x["method"] == "signature_degree_ours"])
    check("ours join error matches frozen result", all(abs(x["join_error_fraction"] - 51 / 5912) < 1e-12 for x in records if x["method"] == "signature_degree_ours"), [x["join_error_fraction"] for x in records if x["method"] == "signature_degree_ours"])
    check("full recomputation reaches zero", all(x["join_error"] == 0 and x["filter_error"] == 0 and x["calls"] == 320 for x in records if x["method"] == "full_recompute"), [(x["calls"], x["join_error"], x["filter_error"]) for x in records if x["method"] == "full_recompute"])
    check("reuse all exposes event impact", all(x["join_error"] == 3095 and x["calls"] == 0 for x in records if x["method"] == "reuse_all"), [(x["calls"], x["join_error"]) for x in records if x["method"] == "reuse_all"])
    check("timings nonnegative", all(x[key] >= 0 for x in records for key in ["prepare_seconds", "selection_seconds", "semantic_seconds", "publish_seconds", "total_seconds"]), "all timing fields")
    check("semantic time positive for called methods", all(x["semantic_seconds"] > 0 for x in records if x["calls"] > 0), [x["semantic_seconds"] for x in records if x["calls"] > 0])
    check("total covers measured parts", all(x["total_seconds"] + 1e-9 >= x["prepare_seconds"] + x["selection_seconds"] + x["semantic_seconds"] + x["publish_seconds"] for x in records), "sum of measured components")
    input_hashes_ok = True
    for item in input_manifest["inputs"].values():
        path = Path(item["path"])
        input_hashes_ok = input_hashes_ok and path.is_file() and sha256(path) == item["sha256"]
    check("input hashes reopen", input_hashes_ok, sorted(input_manifest["inputs"]))
    with (run / "summary.csv").open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    check("summary csv complete", len(csv_rows) == 6 and {x["method"] for x in csv_rows} == expected, [x["method"] for x in csv_rows])
    output_manifest = json.loads((run / "output_manifest.json").read_text(encoding="utf-8"))
    output_hashes_ok = all((run / name).is_file() and sha256(run / name) == item["sha256"] for name, item in output_manifest.items())
    check("recorded output hashes reopen", output_hashes_ok, len(output_manifest))

    status = "VERIFIED" if all(item["passed"] for item in checks) else "FAILED"
    verification = {"status": status, "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks), "checks": checks}
    (run / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if status != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

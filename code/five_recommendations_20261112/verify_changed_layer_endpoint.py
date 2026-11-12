#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    summary_path = args.run_dir / "summary.json"
    records_path = args.run_dir / "records.json"
    manifest_path = args.run_dir / "input_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = summary["summaries"][0]
    checks = {
        "experiment_id": summary["experiment_id"] == "D03-CHANGED-LAYER-ENDPOINT-20261112-V1",
        "three_records": len(records) == 3 and item["repetitions"] == 3,
        "all_calls_145": all(r["calls"] == 145 and r["installed_calls"] == 145 for r in records),
        "zero_stable_calls": all(r["stable_calls"] == 0 for r in records),
        "zero_filter_and_join_error": all(r["filter_error"] == 0 and r["join_error"] == 0 for r in records),
        "positive_denominators": all(r["filter_mass"] > 0 and r["join_mass"] > 0 for r in records),
        "timing_parts_positive": all(
            r["prepare_seconds"] > 0
            and r["selection_seconds"] >= 0
            and r["semantic_seconds"] > 0
            and r["publish_seconds"] > 0
            and r["total_seconds"] >= r["prepare_seconds"] + r["selection_seconds"] + r["semantic_seconds"] + r["publish_seconds"]
            for r in records
        ),
        "input_hashes_match": all(
            Path(v["path"]).is_file() and sha256(Path(v["path"])) == v["sha256"].upper()
            for v in manifest["inputs"].values()
        ),
    }
    output = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "summary_sha256": sha256(summary_path),
        "records_sha256": sha256(records_path),
        "input_manifest_sha256": sha256(manifest_path),
        "median_seconds": {
            "prepare": item["prepare_seconds_median"],
            "selection": item["selection_seconds_median"],
            "semantic": item["semantic_seconds_median"],
            "publish": item["publish_seconds_median"],
            "total": item["total_seconds_median"],
        },
    }
    out = args.run_dir / "verification.json"
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

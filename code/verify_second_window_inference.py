from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--old-pairs", type=Path, required=True)
    parser.add_argument("--revisions", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    old_pairs = json.loads(args.old_pairs.read_text(encoding="utf-8"))
    revisions = json.loads(args.revisions.read_text(encoding="utf-8"))
    manifest = json.loads((args.result_dir / "input_manifest.json").read_text(encoding="utf-8"))
    predictions = json.loads((args.result_dir / "predictions.json").read_text(encoding="utf-8"))
    pair_results = json.loads((args.result_dir / "pair_predicate_results.json").read_text(encoding="utf-8"))
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((args.result_dir / "execution_receipt.json").read_text(encoding="utf-8"))
    old_ids = {int(item["pageid"]) for item in old_pairs["pairs"]}
    new_ids = {int(item["pageid"]) for item in pairs["pairs"]}
    expected_prediction_keys = {
        (int(pair[key]), predicate_index)
        for pair in pairs["pairs"]
        for key in ("old_revid", "new_revid")
        for predicate_index in range(len(contract["semantic_predicates"]))
    }
    prediction_keys = {
        (int(item["revid"]), int(item["predicate_index"])) for item in predictions
    }
    checks = {
        "frozen_before_content": pairs["status"] == "FROZEN_BEFORE_CONTENT_AND_LABEL_ACCESS",
        "metadata_hash_matches": pairs["metadata_sha256"] == contract["metadata_sha256"],
        "old_pairs_hash_matches": contract["excluded_old_pairs_sha256"] == sha256(args.old_pairs),
        "page_ids_disjoint": not (old_ids & new_ids),
        "64_pairs": len(pairs["pairs"]) == 64 and len(new_ids) == 64,
        "revision_content_complete": not revisions["missing_revids"] and revisions["revision_count"] == 128,
        "manifest_contract_hash": manifest["contract"]["sha256"] == sha256(args.contract),
        "manifest_pairs_hash": manifest["pairs"]["sha256"] == sha256(args.pairs),
        "manifest_revisions_hash": manifest["revisions"]["sha256"] == sha256(args.revisions),
        "prediction_keys_complete": prediction_keys == expected_prediction_keys,
        "640_predictions": len(predictions) == 640,
        "320_pair_results": len(pair_results) == 320,
        "receipt_finished": receipt["status"] == "INFERENCE_FINISHED",
        "summary_finished": summary["status"] == "INFERENCE_FINISHED",
        "summary_counts_match": summary["page_pairs"] == 64
        and summary["predicates"] == 5
        and summary["pair_predicates"] == 320,
    }
    output = {
        "status": "VERIFIED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "counts": {
            "page_pairs": len(pairs["pairs"]),
            "predictions": len(predictions),
            "pair_results": len(pair_results),
            "label_flips": summary["label_flips"],
            "pages_with_any_flip": summary["pages_with_any_flip"],
            "visible_input_changes": summary["visible_input_changes"],
        },
        "hashes": {
            "contract": sha256(args.contract),
            "pairs": sha256(args.pairs),
            "old_pairs": sha256(args.old_pairs),
            "revisions": sha256(args.revisions),
            "predictions": sha256(args.result_dir / "predictions.json"),
            "pair_results": sha256(args.result_dir / "pair_predicate_results.json"),
            "summary": sha256(args.result_dir / "summary.json"),
            "receipt": sha256(args.result_dir / "execution_receipt.json"),
        },
        "boundary": "This verifies a second time window with page IDs disjoint from the first. Collection process, model, and predicates remain shared.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

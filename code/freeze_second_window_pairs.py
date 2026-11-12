from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


PREDICATES = [
    "The main subject is a living person.",
    "The main subject is an organization.",
    "The document describes an event that has already occurred.",
    "The document describes a planned or future event.",
    "The document states that the main subject currently holds an official position.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--old-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--target-pairs", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    old_payload = json.loads(args.old_pairs.read_text(encoding="utf-8"))
    old_pageids = {int(item["pageid"]) for item in old_payload["pairs"]}
    by_page = {}
    for row in metadata.get("query", {}).get("recentchanges", []):
        pageid = int(row.get("pageid", 0))
        if (
            pageid > 0
            and pageid not in old_pageids
            and int(row.get("old_revid", 0)) > 0
            and int(row.get("revid", 0)) > 0
        ):
            by_page.setdefault(pageid, row)
    candidates = [by_page[key] for key in sorted(by_page)]
    if len(candidates) < args.target_pairs:
        raise ValueError(
            "fewer disjoint candidate pages than requested: "
            f"{len(candidates)} < {args.target_pairs}"
        )
    selected_indices = sorted(
        random.Random(args.seed).sample(range(len(candidates)), args.target_pairs)
    )
    pairs = []
    for sample_index, index in enumerate(selected_indices):
        row = candidates[index]
        pairs.append(
            {
                "sample_index": sample_index,
                "pageid": int(row["pageid"]),
                "title": row["title"],
                "old_revid": int(row["old_revid"]),
                "new_revid": int(row["revid"]),
                "timestamp": row["timestamp"],
                "oldlen": int(row.get("oldlen", 0)),
                "newlen": int(row.get("newlen", 0)),
            }
        )
    contract = {
        "contract_id": "D03-SECOND-WINDOW-20261110",
        "status": "FROZEN_BEFORE_CONTENT_AND_LABEL_ACCESS",
        "source": "English Wikipedia MediaWiki API recentchanges, main namespace, edit events, non-bot filter",
        "metadata_sha256": sha256(args.metadata),
        "excluded_old_pairs_sha256": sha256(args.old_pairs),
        "selection_seed": args.seed,
        "target_pairs": args.target_pairs,
        "selection_rule": "exclude all page IDs from the 2026-11-01 window; retain one latest observed edit per remaining page; sort by page ID; sample with Python random.Random(seed) without replacement",
        "semantic_predicates": PREDICATES,
        "model": "local Qwen2.5-0.5B-Instruct",
        "decoding": {
            "temperature": 0,
            "max_input_tokens": 768,
            "max_new_tokens": 2,
            "required_output": "0 or 1",
        },
        "scientific_boundaries": [
            "This is a second recent-edit window, not a population estimate.",
            "The 64 page IDs are disjoint from the first window.",
            "The collection process, predicates, and fixed model remain the same.",
            "Model transitions are not human semantic truth.",
            "The sample is frozen before revision content and model labels are read.",
        ],
    }
    frozen = {
        "contract_id": contract["contract_id"],
        "status": contract["status"],
        "metadata_sha256": contract["metadata_sha256"],
        "excluded_old_pairs_sha256": contract["excluded_old_pairs_sha256"],
        "candidate_pages_after_exclusion": len(candidates),
        "excluded_page_count": len(old_pageids),
        "pairs": pairs,
    }
    contract_path = args.output_dir / "contract.json"
    pairs_path = args.output_dir / "frozen_pairs.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    pairs_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "status": "FROZEN_BEFORE_CONTENT_AND_LABEL_ACCESS",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_sha256": sha256(args.metadata),
        "old_pairs_sha256": sha256(args.old_pairs),
        "contract_sha256": sha256(contract_path),
        "pairs_sha256": sha256(pairs_path),
        "candidate_pages_after_exclusion": len(candidates),
        "selected_pairs": len(pairs),
        "overlap_with_old_pageids": len(
            {int(item["pageid"]) for item in pairs} & old_pageids
        ),
    }
    (args.output_dir / "freeze_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "D03-semantic-maintenance-research/0.2 (research artifact)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    frozen = json.loads(args.pairs.read_text(encoding="utf-8"))
    revision_ids = sorted(
        {int(item[key]) for item in frozen["pairs"] for key in ("old_revid", "new_revid")}
    )
    revisions = {}
    batches = []
    for batch_index, start in enumerate(range(0, len(revision_ids), 40)):
        expected = revision_ids[start : start + 40]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "revids": "|".join(map(str, expected)),
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
        }
        request = urllib.request.Request(
            API + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        batch_path = args.output_dir / f"batch_{batch_index:02d}.json"
        batch_path.write_bytes(payload)
        seen = set()
        data = json.loads(payload)
        for page in data.get("query", {}).get("pages", []):
            for revision in page.get("revisions", []):
                revision_id = int(revision["revid"])
                seen.add(revision_id)
                revisions[str(revision_id)] = {
                    "pageid": int(page["pageid"]),
                    "title": page["title"],
                    "revid": revision_id,
                    "parentid": int(revision.get("parentid", 0)),
                    "timestamp": revision["timestamp"],
                    "content": revision.get("slots", {}).get("main", {}).get("content", ""),
                }
        batches.append(
            {
                "path": batch_path.name,
                "sha256": sha256(batch_path),
                "bytes": len(payload),
                "expected": len(expected),
                "received": len(seen),
            }
        )
        time.sleep(0.2)
    missing = sorted(set(map(str, revision_ids)) - set(revisions))
    merged = {
        "pairs_sha256": sha256(args.pairs),
        "revision_count": len(revisions),
        "missing_revids": missing,
        "revisions": revisions,
        "batches": batches,
    }
    merged_path = args.output_dir / "revisions.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "requested": len(revision_ids),
        "received": len(revisions),
        "missing": len(missing),
        "merged_sha256": sha256(merged_path),
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

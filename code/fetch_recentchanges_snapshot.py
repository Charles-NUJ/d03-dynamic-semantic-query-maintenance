from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "D03-semantic-maintenance-research/0.2 (research artifact)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "list": "recentchanges",
        "rcnamespace": "0",
        "rctype": "edit",
        "rcshow": "!bot",
        "rcprop": "title|ids|sizes|flags|timestamp",
        "rclimit": str(args.limit),
    }
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    parsed = json.loads(payload)
    rows = parsed.get("query", {}).get("recentchanges", [])
    if not rows:
        raise ValueError("recentchanges response is empty")
    args.output.write_bytes(payload)
    receipt = {
        "status": "FETCHED_BEFORE_PAIR_SELECTION",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "url_parameters": params,
        "rows": len(rows),
        "output_sha256": sha256(args.output),
    }
    receipt_path = args.output.with_name("recentchanges_fetch_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

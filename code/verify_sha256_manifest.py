from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[".git", "reproduced_results"],
        help="Path relative to root to exclude; may be supplied more than once.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.manifest.resolve()
    excluded = {(root / item).resolve() for item in args.exclude}

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == item or item in resolved.parents for item in excluded)

    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / Path(relative)).resolve()
        inside = path == root or root in path.parents
        exists = path.is_file()
        actual = sha256(path) if inside and exists else None
        entries.append(
            {
                "path": relative,
                "inside_root": inside,
                "exists": exists,
                "expected": expected,
                "actual": actual,
                "match": inside and exists and actual == expected,
            }
        )

    manifest_names = {"SHA256SUMS.txt", "PACKAGE_SHA256_MANIFEST_20261112.txt"}
    current_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not is_excluded(path)
        and path.relative_to(root).as_posix() not in manifest_names
    }
    registered_files = {entry["path"] for entry in entries}
    output = {
        "status": "VERIFIED"
        if entries
        and all(entry["match"] for entry in entries)
        and current_files == registered_files
        else "FAILED",
        "entry_count": len(entries),
        "all_hashes_match": bool(entries) and all(entry["match"] for entry in entries),
        "unregistered_files": sorted(current_files - registered_files),
        "missing_registered_files": sorted(registered_files - current_files),
        "manifest_sha256": sha256(manifest),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if output["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

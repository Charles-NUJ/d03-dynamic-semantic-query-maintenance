from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path relative to root to exclude; may be supplied more than once.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    excluded = {output}
    excluded.update((root / item).resolve() for item in args.exclude)

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == item or item in resolved.parents for item in excluded)

    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not is_excluded(path)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )
    print(f"WROTE {output} WITH {len(files)} ENTRIES")


if __name__ == "__main__":
    main()

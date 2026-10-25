"""Check the seven Qwen files used by the recorded experiment."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parents[1]
entries = json.loads((root / "reference/model_files.json").read_text(encoding="utf-8-sig"))
for entry in entries:
    path = root / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Model file SHA-256 mismatch: {path.name}")
    print(f"OK {path.name}")
print(f"Checked {len(entries)} model files.")
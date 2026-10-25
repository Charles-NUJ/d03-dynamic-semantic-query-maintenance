"""Restore saved inference returns for the Financial PhraseBank replay."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parents[1]
source = root / "reference/financial_prediction_returns"
target = root / "raw_results/major_p_fpb_v1"
if not (target / "prepared.json").is_file():
    raise RuntimeError("Run the financial prepare command before restoring predictions.")
items = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
for entry in items:
    name = entry["name"]
    if Path(name).name != name:
        raise ValueError("Unexpected prediction-file name.")
    src, dst = source / name, target / name
    if dst.exists():
        raise FileExistsError(dst)
    if hashlib.sha256(src.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {name}")
for entry in items:
    shutil.copyfile(source / entry["name"], target / entry["name"])
print("Restored four hash-checked historical inference files; no new model calls.")
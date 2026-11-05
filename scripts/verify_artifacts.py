"""Verify checked-in bytes without unpickling the model."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "results/manifest.json").read_text(encoding="utf-8"))
for section in ("sha256", "source_sha256"):
    for path, expected in manifest[section].items():
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"hash mismatch: {path}")
print(f"Verified {sum(map(len, manifest.values()))} artifact/source hashes")

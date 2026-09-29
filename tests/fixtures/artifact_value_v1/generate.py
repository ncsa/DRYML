"""Generate fixed trusted U4 Value v1 payload fixtures without using Value."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import dill
import numpy as np


ROOT = Path(__file__).parent
FORMAT = "dryml.artifacts.value"
FIXTURES = {
    "absent": None,
    "none": None,
    "float": 3.5,
    "array": np.array([1.0, 2.5], dtype=np.float64),
    "matrix": np.array([[1, 2], [3, 4]], dtype=np.int64),
    "tree": {"labels": ("a", "b"), "scores": [0.25, 0.75]},
}


def main() -> None:
    """Write fixed envelopes and a provenance manifest with their byte hashes."""

    entries = {}
    for name, result in FIXTURES.items():
        present = name != "absent"
        payload = {
            "format": FORMAT,
            "version": 1,
            "present": present,
            "result": result if present else None,
        }
        directory = ROOT / name
        directory.mkdir(exist_ok=True)
        output = directory / "value.pkl"
        output.write_bytes(dill.dumps(payload, protocol=5))
        entries[name] = {
            "file": f"{name}/value.pkl",
            "sha256": sha256(output.read_bytes()).hexdigest(),
            "present": present,
        }
    (ROOT / "manifest.json").write_text(json.dumps({
        "format": FORMAT,
        "version": 1,
        "generator": "generate.py",
        "provenance": "Hand-authored fixed U4 beta cases; envelopes are encoded directly with dill protocol 5, not Value.",
        "fixtures": entries,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

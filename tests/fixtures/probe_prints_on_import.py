"""Probe fixture that writes output during import."""

from __future__ import annotations

import sys

print("probe fixture stdout")
print("probe fixture stderr", file=sys.stderr)


def noisy_target():
    return "not executed"

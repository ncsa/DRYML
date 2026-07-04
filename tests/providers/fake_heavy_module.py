"""Sentinel module that must only be imported in probe children."""

import os

IMPORTED = True
os.environ["DRYML_FAKE_HEAVY_IMPORTED"] = "1"

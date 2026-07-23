"""Probe fixture that sleeps during import for timeout tests."""

from __future__ import annotations

import time

time.sleep(5)


def slow_target():
    return "not executed"

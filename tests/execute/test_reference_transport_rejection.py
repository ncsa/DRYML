import os
from pathlib import Path
import subprocess
import sys


_SCRIPT = r'''
import sys
from dryml.execute._spooling import serialize_call

class CoreSemanticMarker:
    __module__ = "dryml.core.synthetic"

class Envelope:
    def __init__(self, value):
        self.value = value

try:
    serialize_call(lambda value: value, (Envelope(CoreSemanticMarker()),), {}, limit_bytes=1_000_000)
except TypeError as error:
    assert str(error) == "live resource is unsupported by Execute transport"
else:
    raise AssertionError("core semantic value was accepted")
assert "dryml.core" not in sys.modules
'''


def test_serializer_rejects_core_marked_values_without_loading_core():
    """Generic preflight rejects retired core values without importing core itself."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Dependency-light boundary facts and capture-on-request execution contracts."""

import subprocess
import sys

import pytest

from dryml.methods.stream import StreamGraphCursor, _StreamOp
from dryml.methods.stream_state import PullKind, StreamExecution


class Binding:
    """Inert adapter fixture exposing the kernel source-opening protocol."""

    def __init__(self, values):
        self.values = values
        self.opens = 0

    def open(self, epoch):
        self.opens += 1
        return iter(self.values)


def test_optional_pull_interface_preserves_native_block_identity_and_root_eof():
    block = object()
    binding = Binding((block,))
    cursor: StreamExecution = StreamGraphCursor(_StreamOp("source", source=binding))
    assert binding.opens == 0
    with pytest.raises(RuntimeError, match="delivery"):
        _ = cursor.value
    assert cursor.pull() is PullKind.DELIVERY
    assert cursor.value is block
    assert binding.opens == 1
    assert cursor.pull() is PullKind.ROOT_END
    assert cursor.pull() is PullKind.ROOT_END
    assert cursor.observed_eof
    with pytest.raises(RuntimeError, match="delivery"):
        _ = cursor.value


def test_explicit_close_does_not_masquerade_as_root_completion():
    binding = Binding((1,))
    cursor = StreamGraphCursor(_StreamOp("source", source=binding))
    cursor.close()
    with pytest.raises(RuntimeError, match="root-completion"):
        cursor.pull()
    assert binding.opens == 0
    assert not cursor.observed_eof


def test_optional_pull_failure_is_not_a_completion_signal():
    class FailedBinding:
        def open(self, epoch):
            raise ValueError("acquisition failed")

    cursor = StreamGraphCursor(_StreamOp("source", source=FailedBinding()))
    with pytest.raises(ValueError, match="acquisition failed"):
        cursor.pull()
    assert not cursor.observed_eof


def test_section_end_requires_explicit_handoff_and_is_not_root_end():
    """A native executor can supply boundary facts without per-element wrappers."""
    class NativeSections:
        value = object()
        pending = False
        finished = False

        def pull(self):
            if self.finished:
                return PullKind.ROOT_END
            self.pending = True
            return PullKind.SECTION_END

        def handoff(self):
            if not self.pending:
                raise RuntimeError("no pending boundary")
            self.pending = False
            self.finished = True

    execution = NativeSections()
    with pytest.raises(RuntimeError, match="pending"):
        execution.handoff()
    assert execution.pull() is execution.pull() is PullKind.SECTION_END
    execution.handoff()
    assert execution.pull() is PullKind.ROOT_END


def test_fresh_stream_state_import_and_execution_stay_below_consumers():
    script = """
import sys
import dryml.methods.stream_state
from dryml.methods.stream import StreamGraphCursor, _StreamOp
class Binding:
    def open(self, epoch):
        return iter((1, 2))
assert list(StreamGraphCursor(_StreamOp('source', source=Binding()))) == [1, 2]
for prefix in ('dryml.data', 'dryml.models', 'dryml.artifacts', 'tensorflow', 'torch', 'jax', 'ray'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("backing", (
    "bytes", "bytes-memoryview", "numpy-owned", "numpy-memoryview",
    "writable", "writable-base", "bytearray", "readonly-memoryview", "foreign",
))
def test_retained_numpy_buffer_shares_only_stable_backing(backing):
    """Readonly flags do not protect storage writable through another owner."""
    import numpy as np
    from dryml.methods.stream_state import _retain_buffer

    original = np.arange(4, dtype=np.int64)
    safe = backing in ("bytes", "bytes-memoryview", "numpy-owned", "numpy-memoryview")
    if backing in ("bytes", "bytes-memoryview"):
        owner = original.tobytes()
        value = np.frombuffer(owner if backing == "bytes" else memoryview(owner), dtype=np.int64)
    elif backing in ("numpy-owned", "numpy-memoryview"):
        original.flags.writeable = False
        value = original if backing == "numpy-owned" else np.frombuffer(memoryview(original), dtype=np.int64)
    elif backing in ("bytearray", "readonly-memoryview"):
        owner = bytearray(original.tobytes())
        view = memoryview(owner)
        value = np.frombuffer(view if backing == "bytearray" else view.toreadonly(), dtype=np.int64)
        value.flags.writeable = False
    elif backing == "foreign":
        class ForeignOwner:
            __array_interface__ = dict(original.__array_interface__,
                                       data=(original.ctypes.data, True))
        value = np.asarray(ForeignOwner())
    else:
        value = original if backing == "writable" else original.view()
        if backing == "writable-base":
            value.flags.writeable = False
    retained = _retain_buffer(value)
    assert (retained is value) is safe
    if not safe:
        assert not np.shares_memory(retained, value)
        if backing in ("bytearray", "readonly-memoryview"):
            owner[:] = np.full(4, -1, dtype=np.int64).tobytes()
        else:
            original[:] = -1
    np.testing.assert_array_equal(retained, np.arange(4))

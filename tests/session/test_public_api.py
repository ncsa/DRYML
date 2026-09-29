"""Public import and passive-observation contracts for ``dryml.session``."""

import subprocess
import sys


def test_session_is_a_lazy_root_export():
    """The root package exposes the distinct public session facade lazily."""

    import dryml

    assert dryml.session.mode() == "python"


def test_fresh_inspection_is_passive_and_public_exports_exclude_workers():
    """Inspection performs no host probe and deferred worker names stay absent."""

    script = """
import sys
import dryml.session.state as state
state.local_inventory = lambda: (_ for _ in ()).throw(AssertionError('inventory probe'))
state.inspect_current = lambda: (_ for _ in ()).throw(AssertionError('environment probe'))
from dryml import session
assert session.current().mode == 'python'
assert session.mode() == 'python'
assert not {'tensorflow', 'torch', 'jax', 'jaxlib'} & set(sys.modules)
for name in ('worker_env_request', 'worker_world_request', 'publish_worker_session'):
    assert not hasattr(session, name)
from dryml.runtime import RuntimeMode
assert set(RuntimeMode.__members__) == {'NONE', 'ORCHESTRATOR', 'INLINE'}
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

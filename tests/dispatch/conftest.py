import sys
import textwrap

import pytest


@pytest.fixture()
def target_module(tmp_path):
    module = tmp_path / "dispatch_target.py"
    module.write_text(
        textwrap.dedent(
            '''
            import os
            import signal
            import sys
            import time

            from dryml.core2.object import Pickleable
            from dryml.runtime import BOOTSTRAP_MARKER_ENV, active_runtime, active_runtime_mode

            IMPORT_RUNTIME_MODE = active_runtime_mode().value

            class Box(Pickleable):
                def __init__(self, value):
                    super().__init__()
                    self.value = value

                def plus(self, amount):
                    return self.value + amount

            def add(x, y):
                return x + y

            def noisy_add(x, y):
                print("hello stdout")
                print("hello stderr", file=sys.stderr)
                return x + y

            def runtime_status():
                return {"mode": active_runtime_mode().value, "bootstrap": os.environ.get(BOOTSTRAP_MARKER_ENV), "import_mode": IMPORT_RUNTIME_MODE}

            def allocation_facts():
                runtime = active_runtime()
                alloc = runtime.allocation
                return {
                    "mode": runtime.mode.value,
                    "role": alloc.role,
                    "replica": alloc.replica,
                    "rank": alloc.rank,
                    "local_rank": alloc.local_rank,
                    "cpus": list(alloc.cpus),
                    "accelerators": {key: list(value) for key, value in alloc.accelerators.items()},
                    "world_allocation_id": alloc.world_allocation_id,
                    "env_role": os.environ.get("DRYML_WORLD_ROLE"),
                    "env_replica": os.environ.get("DRYML_WORLD_REPLICA"),
                    "env_rank": os.environ.get("DRYML_WORLD_RANK"),
                    "env_local_rank": os.environ.get("DRYML_WORLD_LOCAL_RANK"),
                    "env_world_allocation_id": os.environ.get("DRYML_WORLD_ALLOCATION_ID"),
                    "is_no_allocation": getattr(alloc, "is_no_allocation", None),
                    "import_mode": IMPORT_RUNTIME_MODE,
                }

            def fail_for_role(role):
                if os.environ.get("DRYML_WORLD_ROLE") == role:
                    raise ValueError(f"failed role {role}")
                time.sleep(5)
                return allocation_facts()

            def box_value(box):
                return box.value

            def ref_value(ref):
                return ref

            def make_box(value):
                return Box(value)

            def fail():
                print("before failure")
                raise ValueError("expected dispatch failure")

            def sleep_forever():
                signal.signal(signal.SIGINT, lambda signum, frame: time.sleep(10))
                while True:
                    time.sleep(0.1)
            '''
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        yield module
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("dispatch_target", None)

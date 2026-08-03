import sys
import textwrap

import pytest


@pytest.fixture()
def target_module(tmp_path):
    module = tmp_path / "dispatch_target.py"
    fake_torch = tmp_path / "torch.py"
    fake_torch.write_text(
        textwrap.dedent(
            '''
            import os

            MARKER = "fake-dispatch-torch"
            THREADS = None
            IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

            class cuda:
                @staticmethod
                def device_count():
                    return len([value for value in (IMPORT_CUDA_VISIBLE_DEVICES or "").split(",") if value])

            def set_num_threads(value):
                global THREADS
                THREADS = value
            '''
        ),
        encoding="utf-8",
    )
    module.write_text(
        textwrap.dedent(
            '''
            import os
            import signal
            import sys
            import time

            import dryml
            from dryml.core.object import Pickleable
            from dryml.managed import ManagedOutput, OperationResult, current_operation_context, managed
            from dryml.records import make_representation_spec
            from dryml.runtime import BOOTSTRAP_MARKER_ENV, active_runtime, active_runtime_mode, enforcement, import_configured_framework

            IMPORT_RUNTIME_MODE = active_runtime_mode().value

            class Box(Pickleable):
                def __init__(self, value):
                    super().__init__()
                    self.value = value

                def plus(self, amount):
                    return self.value + amount

            MANAGED_REPRESENTATION = make_representation_spec(
                "dispatch.managed.bytes",
                version="1",
                storage_kinds=("product-dir",),
            )

            class ManagedBox(Pickleable):
                @managed(
                    outputs=(ManagedOutput("result", primary=True, kind="data"),),
                    resumable=True,
                    checkpoint_schema="dispatch-checkpoint-v1",
                )
                @dryml.env.req(python=">=3")
                @dryml.world.req(cpus={"exact": 1})
                def compute(
                    self,
                    value="value",
                    fail=False,
                    sleep=0.0,
                    hard_exit=False,
                ):
                    context = current_operation_context()
                    runtime = active_runtime()
                    context.progress(
                        1,
                        total=2,
                        message=runtime.mode.value,
                        metrics={"rank": runtime.allocation.rank},
                    )
                    if sleep:
                        time.sleep(sleep)

                    def checkpoint():
                        context.write_checkpoint("cursor.txt", (b"1",))

                    context.safe_point(checkpoint=checkpoint)
                    if hard_exit:
                        os._exit(17)
                    payload = f"{value}:{runtime.mode.value}:{runtime.allocation.rank}".encode()
                    context.write_output(
                        "result",
                        "value.bin",
                        (payload,),
                        representation=MANAGED_REPRESENTATION,
                    )
                    if fail:
                        message = fail if isinstance(fail, str) else "managed worker failure"
                        raise RuntimeError(message)
                    context.progress(2, total=2, message="complete")
                    return OperationResult()

            class ManagedConsumer(Pickleable):
                def __init__(self, source):
                    super().__init__()
                    self.source = source

                def __dryml_managed_inputs__(self, method, args, kwargs):
                    return (self.source,)

                @managed(
                    outputs=(ManagedOutput("result", primary=True, kind="data"),),
                )
                def compute(self):
                    context = current_operation_context()
                    context.write_output(
                        "result",
                        "value.bin",
                        (b"consumed",),
                        representation=MANAGED_REPRESENTATION,
                    )

            def add(x, y):
                return x + y

            def noisy_add(x, y):
                print("hello stdout")
                print("hello stderr", file=sys.stderr)
                return x + y

            def runtime_status():
                snapshot = dryml.session.current()
                return {
                    "mode": active_runtime_mode().value,
                    "bootstrap": os.environ.get(BOOTSTRAP_MARKER_ENV),
                    "import_mode": IMPORT_RUNTIME_MODE,
                    "enforcement": enforcement().value,
                    "selected_environment": None if snapshot.selected_environment is None else snapshot.selected_environment.kind,
                    "selected_world": None if snapshot.selected_world is None else sorted(snapshot.selected_world.roles),
                    "selected_runtime": None if snapshot.selected_runtime is None else snapshot.selected_runtime.mode.value,
                    "compatibility_policy": snapshot.compatibility_policy,
                    "compatibility_axes": None if snapshot.compatibility_axes is None else snapshot.compatibility_axes.to_data(),
                }

            def configured_torch_import_status():
                module = import_configured_framework("torch")
                return {
                    "mode": active_runtime_mode().value,
                    "bootstrap": os.environ.get(BOOTSTRAP_MARKER_ENV),
                    "marker": getattr(module, "MARKER", None),
                    "threads": getattr(module, "THREADS", None),
                    "cuda_visible_devices": getattr(module, "IMPORT_CUDA_VISIBLE_DEVICES", None),
                }

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

            def argument_values(box, ref, literal):
                return [box.value, ref, literal]

            def make_box(value):
                return Box(value)

            def fail(message="expected dispatch failure"):
                print("before failure")
                raise ValueError(message)

            def fail_secret():
                raise ValueError("dispatch-secret-sentinel-91e6")

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

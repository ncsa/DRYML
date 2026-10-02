"""Real-kernel checks for notebook Dispatch and shell task ownership."""

from __future__ import annotations

import asyncio
import os

import pytest


def test_repeated_notebook_dispatch_preserves_shell_and_kernel_task(tmp_path):
    """Successful publications leave later cells and unrelated kernel tasks alive."""

    pytest.importorskip("ipykernel")
    client_module = pytest.importorskip("jupyter_client")

    async def exercise():
        manager = client_module.AsyncKernelManager(kernel_name="python3")
        await manager.start_kernel(env={**os.environ, "DRYML_NOTEBOOK_TEST_ROOT": str(tmp_path)})
        client = manager.client()
        client.start_channels()
        try:
            await client.wait_for_ready(timeout=30)

            async def execute(source, *, expected="ok", interrupt=False):
                message_id = client.execute(source)
                if interrupt:
                    while True:
                        busy = await asyncio.wait_for(client.get_iopub_msg(), timeout=30)
                        if busy["parent_header"].get("msg_id") == message_id and busy["msg_type"] == "status" and busy["content"]["execution_state"] == "busy":
                            break
                    await asyncio.sleep(2)
                    await manager.interrupt_kernel()
                while True:
                    reply = await asyncio.wait_for(client.get_shell_msg(), timeout=60)
                    if reply["parent_header"].get("msg_id") == message_id:
                        break
                output = []
                while True:
                    message = await asyncio.wait_for(client.get_iopub_msg(), timeout=60)
                    if message["parent_header"].get("msg_id") != message_id:
                        continue
                    if message["msg_type"] == "stream":
                        output.append(message["content"]["text"])
                    elif message["msg_type"] == "error":
                        output.extend(message["content"]["traceback"])
                    elif message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                        assert reply["content"]["status"] == expected, "\n".join(output)
                        if expected == "error":
                            assert reply["content"]["ename"] == "KeyboardInterrupt", "\n".join(output)
                        return "\n".join(output)

            await execute(
                "import asyncio, os\n"
                "from pathlib import Path\n"
                "from dryml import dispatch\n"
                "from dryml.core import Repo, StateRef\n"
                "from dryml.core.execute import CoreOptions\n"
                "from dryml.core.execute_codec import CoreCallCodecError\n"
                "from dryml.core.store.dir import DirStore\n"
                "from dryml.execute.subprocess import SubProcessConfig\n"
                "from tests.dispatch.test_backend_execution import _StatefulResult\n"
                "root = Path(os.environ['DRYML_NOTEBOOK_TEST_ROOT'])\n"
                "(root / 'spool').mkdir()\n"
                "repo = Repo(DirStore(root / 'store'))\n"
                "view = dispatch.with_options(backend=SubProcessConfig(spool_directory=root / 'spool'), core=CoreOptions(repo=repo, return_objects=False))\n"
                "def notebook_fn(value):\n    return _StatefulResult(value)\n"
                "kernel_task = asyncio.create_task(asyncio.Event().wait())\n"
                "initial_shell_task = asyncio.current_task()\n"
            )
            shell_status = (
                "print('shell-alive', not kernel_task.done(), "
                "kernel_task in asyncio.all_tasks(), initial_shell_task.done(), "
                "sum(t.get_coro().__qualname__ == 'Kernel.shell_main' "
                "for t in asyncio.all_tasks()))"
            )
            for count in range(1, 4):
                result = await execute(
                    f"result = view.run(notebook_fn, {count})\n"
                    f"print('published', isinstance(result, StateRef), len(list(repo.find_defs(None, refresh=True))))\n"
                )
                assert f"published True {count}" in result
                assert "DispatchCoverageWarning" not in result
                assert "Task was destroyed" not in result
                alive = await execute(shell_status)
                assert "shell-alive True True True 1" in alive
                assert "Kernel.shell_main" not in alive
            rejected = await execute(
                "def captured_kernel_task():\n    return kernel_task\n"
                "try:\n    view.run(captured_kernel_task)\n"
                "except CoreCallCodecError:\n    print('task-transport-rejected', not kernel_task.done())\n"
            )
            assert "task-transport-rejected True" in rejected
            assert "Task was destroyed" not in rejected
            assert "shell-alive True True True 1" in await execute(shell_status)
            interrupted = await execute(
                "def slow_notebook_fn():\n    import time\n    time.sleep(30)\n    return _StatefulResult(99)\n"
                "view.run(slow_notebook_fn)",
                expected="error",
                interrupt=True,
            )
            assert "KeyboardInterrupt" in interrupted
            after = await execute(
                "print('after-interrupt', len(list(repo.find_defs(None, refresh=True))), not kernel_task.done())"
            )
            assert "after-interrupt 3 True" in after
            await execute("kernel_task.cancel()")
        finally:
            client.stop_channels()
            await manager.shutdown_kernel(now=True)

    asyncio.run(exercise())

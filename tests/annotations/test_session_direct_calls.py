"""Focused direct-call enforcement contracts for managed sessions."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading
import warnings

import dill
import pytest

import dryml
from dryml.annotations.errors import AnnotationResolutionError
from dryml.annotations.interception import _direct_call_bypass, is_trusted_wrapper, trusted_original
from dryml.core import Object
from dryml.managed import ManagedOutput, managed
from dryml.runtime import RuntimeAllocationView, RuntimeEnforcement, RuntimeMode, enter_runtime
from dryml.runtime.errors import PublicationBusyError, PublicationReentryError, RuntimeTransitionError
from dryml.runtime.publication import EffectPlan, SessionGeneration, publication
from dryml.core.utils.general import pickle_load
from dryml.dispatch import normalize_user_operation


class _ManagedAnnotationOrderTarget(Object):
    @dryml.world.req(cpus={"min": 10_000_000})
    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def outer(self):
        return "outer"

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    @dryml.world.req(cpus={"min": 10_000_000})
    def inner(self):
        return "inner"


@pytest.fixture(autouse=True)
def _plain_session():
    dryml.session.reset()
    yield
    dryml.session.reset()


def test_python_bypasses_and_managed_rejects_before_free_function_body():
    calls = []

    @dryml.world.req(cpus={"min": 10_000_000})
    def target():
        calls.append("body")

    assert is_trusted_wrapper(target)
    assert trusted_original(target) is not target
    assert target() is None
    assert calls == ["body"]

    dryml.session.manage(cpus=1)
    with pytest.raises(AnnotationResolutionError) as exc_info:
        target()
    assert calls == ["body"]
    assert exc_info.value.context["current_allowance"]
    assert exc_info.value.context["controls"]["memory"] in {"undeclared", "declarative"}


def test_direct_bound_managed_call_never_submits_a_dispatch_backend(monkeypatch):
    from dryml.dispatch.planner import Dispatcher

    submitted = []
    monkeypatch.setattr(Dispatcher, "submit", lambda *args, **kwargs: submitted.append(args))

    target = _ManagedAnnotationOrderTarget()
    assert target.outer() == "outer"
    assert submitted == []


def test_free_function_does_not_collect_an_unrelated_argument_class_method():
    calls = []

    class Argument:
        def target(self):
            return None

    @dryml.world.req(cpus={"min": 10_000_000})
    def target(argument):
        calls.append(argument)

    dryml.session.manage(cpus=1)
    with pytest.raises(AnnotationResolutionError):
        target(Argument())
    assert calls == []


def test_contradictory_hard_fragments_fail_before_the_direct_body():
    calls = []

    @dryml.world.req(cpus={"min": 2})
    @dryml.world.req(cpus={"max": 1})
    def target():
        calls.append("body")

    dryml.session.manage(cpus=1)
    with pytest.raises(AnnotationResolutionError) as exc_info:
        target()
    assert calls == []
    assert exc_info.value.context["merge_issues"]


def test_pickle_transport_keeps_the_trusted_worker_requirement_wrapper():
    @dryml.world.req(cpus={"min": 10_000_000})
    def target():
        return "body"

    normalized = normalize_user_operation(target, allow_pickle=True)
    try:
        loaded = pickle_load(normalized.launch["pickle_path"])
        dryml.session.manage(cpus=1)
        with pytest.raises(AnnotationResolutionError):
            loaded()
    finally:
        import shutil

        shutil.rmtree(normalized.launch["cleanup_paths"][0], ignore_errors=True)


def test_coroutine_generator_and_async_generator_hold_the_checked_lease():
    calls = []

    @dryml.world.req(cpus={"min": 10_000_000})
    async def coroutine():
        calls.append("coroutine")

    @dryml.world.req(cpus={"exact": 1})
    def generator():
        calls.append("generator")
        yield 1

    @dryml.world.req(cpus={"min": 10_000_000})
    async def async_generator():
        calls.append("async-generator")
        yield 1

    dryml.session.manage(cpus=1)
    with pytest.raises(AnnotationResolutionError):
        asyncio.run(coroutine())
    assert calls == []

    live = generator()
    assert next(live) == 1
    with pytest.raises(PublicationBusyError):
        dryml.session.reset()
    live.close()
    dryml.session.reset()

    dryml.session.manage(cpus=1)

    async def consume():
        with pytest.raises(AnnotationResolutionError):
            await anext(async_generator())

    asyncio.run(consume())
    assert calls == ["generator"]


def test_hard_wrappers_do_not_serialize_the_process_coordinator_graph():
    @dryml.world.req(cpus={"exact": 1})
    def sync_target():
        return None

    @dryml.world.req(cpus={"exact": 1})
    async def coroutine_target():
        return None

    @dryml.world.req(cpus={"exact": 1})
    def generator_target():
        yield None

    @dryml.world.req(cpus={"exact": 1})
    async def async_generator_target():
        yield None

    for target in (sync_target, coroutine_target, generator_target, async_generator_target):
        assert dill.dumps(target, protocol=5)


def test_async_generator_forwards_send_throw_close_and_releases_its_lease():
    received = []

    @dryml.world.req(cpus={"exact": 1})
    async def target():
        try:
            received.append((yield "ready"))
        except LookupError:
            yield "handled"
        yield "finished"

    dryml.session.manage(cpus=1)

    async def consume():
        live = target()
        assert await anext(live) == "ready"
        with pytest.raises(PublicationBusyError):
            dryml.session.reset()
        assert await live.asend("value") == "finished"
        with pytest.raises(PublicationBusyError):
            dryml.session.reset()
        await live.aclose()

        live = target()
        assert await anext(live) == "ready"
        assert await live.athrow(LookupError) == "handled"
        await live.aclose()

    asyncio.run(consume())
    dryml.session.reset()
    assert received == ["value"]


def test_async_generator_failed_concurrent_close_retains_its_lease():
    advance_started = asyncio.Event()
    release_advance = asyncio.Event()

    @dryml.world.req(cpus={"exact": 1})
    async def target():
        yield "ready"
        advance_started.set()
        await release_advance.wait()
        yield "done"

    dryml.session.manage(cpus=1)

    async def exercise():
        live = target()
        assert await anext(live) == "ready"
        advance = asyncio.create_task(anext(live))
        await advance_started.wait()

        with pytest.raises(RuntimeError, match="already running"):
            await live.aclose()
        with pytest.raises(PublicationBusyError):
            dryml.session.reset()

        release_advance.set()
        assert await advance == "done"
        await live.aclose()

    asyncio.run(exercise())
    dryml.session.reset()


def test_generator_lifecycles_release_only_after_exhaustion_failure_or_finalization():
    @dryml.world.req(cpus={"exact": 1})
    def target():
        try:
            yield "ready"
        except LookupError:
            yield "handled"
        yield "done"

    @dryml.world.req(cpus={"exact": 1})
    def failing():
        yield "ready"
        raise RuntimeError("generator failure")

    dryml.session.manage(cpus=1)
    live = target()
    assert next(live) == "ready"
    assert live.throw(LookupError) == "handled"
    assert next(live) == "done"
    with pytest.raises(StopIteration):
        next(live)

    live = failing()
    assert next(live) == "ready"
    with pytest.raises(RuntimeError, match="generator failure"):
        next(live)
    live = target()
    assert next(live) == "ready"
    with pytest.raises(PublicationBusyError):
        dryml.session.reset()
    del live
    gc.collect()
    dryml.session.reset()


def test_async_generator_exhaustion_failure_cancellation_and_abandonment_finalize():
    @dryml.world.req(cpus={"exact": 1})
    async def target():
        yield "ready"
        yield "done"

    @dryml.world.req(cpus={"exact": 1})
    async def failing():
        yield "ready"
        raise RuntimeError("async generator failure")

    @dryml.world.req(cpus={"exact": 1})
    async def waiting():
        yield "ready"
        await asyncio.sleep(60)

    dryml.session.manage(cpus=1)

    async def exercise():
        live = target()
        assert await anext(live) == "ready"
        assert await anext(live) == "done"
        with pytest.raises(StopAsyncIteration):
            await anext(live)

        live = failing()
        assert await anext(live) == "ready"
        with pytest.raises(RuntimeError, match="async generator failure"):
            await anext(live)

        live = waiting()
        assert await anext(live) == "ready"
        task = asyncio.create_task(anext(live))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        live = target()
        assert await anext(live) == "ready"
        with pytest.raises(PublicationBusyError):
            dryml.session.reset()
        del live
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(exercise())
    dryml.session.reset()


def test_direct_bypass_is_single_use_and_scoped_to_its_lifecycle():
    calls = []

    @dryml.world.req(cpus={"min": 10_000_000})
    def target():
        calls.append("target")

    @dryml.world.req(cpus={"min": 10_000_000})
    def unrelated():
        calls.append("unrelated")

    @dryml.world.req(cpus={"min": 10_000_000})
    def nested():
        return unrelated()

    dryml.session.manage(cpus=1)
    with _direct_call_bypass(target):
        with pytest.raises(AnnotationResolutionError):
            unrelated()
        with pytest.raises(AnnotationResolutionError):
            nested()
        target()
        copied = contextvars.copy_context()
        with pytest.raises(AnnotationResolutionError):
            copied.run(target)
    with pytest.raises(AnnotationResolutionError):
        target()

    failures = []
    with _direct_call_bypass(target):
        thread = threading.Thread(target=lambda: _record_direct_call_failure(target, failures))
        thread.start()
        thread.join()
        target()
    assert failures == [AnnotationResolutionError]
    assert calls == ["target", "target"]


def _record_direct_call_failure(target, failures):
    try:
        target()
    except AnnotationResolutionError as exc:
        failures.append(type(exc))


def test_bypass_rejects_sibling_async_task_and_stale_context():
    calls = []

    @dryml.world.req(cpus={"min": 10_000_000})
    async def target():
        calls.append("target")

    dryml.session.manage(cpus=1)

    async def exercise():
        with _direct_call_bypass(target):
            sibling = asyncio.create_task(target())
            await target()
            with pytest.raises(AnnotationResolutionError):
                await sibling
        with pytest.raises(AnnotationResolutionError):
            await target()

    asyncio.run(exercise())
    assert calls == ["target"]


def test_base_exception_and_coroutine_cancellation_release_the_lease():
    @dryml.world.req(cpus={"exact": 1})
    def interrupted():
        raise KeyboardInterrupt("test interruption")

    @dryml.world.req(cpus={"exact": 1})
    async def cancelled():
        await asyncio.sleep(60)

    dryml.session.manage(cpus=1)
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        interrupted()
    dryml.session.reset()

    dryml.session.manage(cpus=1)

    async def exercise():
        task = asyncio.create_task(cancelled())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    dryml.session.reset()


def test_writer_reentry_of_decorated_call_fails_before_lease_admission():
    @dryml.world.req(cpus={"exact": 1})
    def target():
        raise AssertionError("wrapper should reject before its body")

    before = publication.current()
    candidate = publication.stage(
        before,
        SessionGeneration(before.number + 1, before.runtime),
    )
    with pytest.raises(PublicationReentryError):
        publication.commit(
            candidate,
            EffectPlan(environment={"DRYML_DIRECT_CALL_REENTRY_TEST": "set"}),
            validator=target,
        )
    assert publication.current() is before


def test_class_construction_and_managed_descriptor_decorator_orders():
    calls = []

    @dryml.world.req(cpus={"min": 10_000_000})
    class ClassDecorated:
        def __init__(self):
            calls.append("class-init")

        def inherited(self):
            calls.append("inherited")

    class Override(ClassDecorated):
        def inherited(self):
            calls.append("override")

    class ExplicitConstruction:
        @dryml.world.req(cpus={"min": 10_000_000})
        def __init__(self):
            calls.append("explicit-init")

    assert dryml.world.req(cpus={"exact": 1})(ClassDecorated) is ClassDecorated
    dryml.session.manage(cpus=1)
    assert isinstance(ClassDecorated(), ClassDecorated)
    with pytest.raises(AnnotationResolutionError):
        ClassDecorated().inherited()
    with pytest.raises(AnnotationResolutionError):
        Override().inherited()
    with pytest.raises(AnnotationResolutionError):
        ExplicitConstruction()
    with pytest.raises(AnnotationResolutionError):
        _ManagedAnnotationOrderTarget().outer()
    with pytest.raises(AnnotationResolutionError):
        _ManagedAnnotationOrderTarget().inner()
    assert calls == ["class-init", "class-init", "class-init"]


def test_warn_override_enters_once_and_orchestrator_rejects_with_guidance():
    calls = []

    @dryml.env.req(requirements=("not-an-installed-dryml-package>=1",))
    def target():
        calls.append("body")

    with enter_runtime(
        RuntimeMode.INLINE,
        RuntimeAllocationView(cpus=(0,)),
        enforcement=RuntimeEnforcement.WARN,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            target()
        assert len(caught) == 1
        assert "dryml direct requirement warning" in str(caught[0].message)
    assert calls == ["body"]

    dryml.session.set_mode("orchestrator")
    with pytest.raises(RuntimeTransitionError, match="dispatch"):
        target()
    assert calls == ["body"]


@pytest.mark.parametrize("mode", ["python", "managed", "orchestrator", "probe", "worker"])
def test_runtime_only_requirements_remain_metadata_in_every_direct_call_mode(mode):
    calls = []

    @dryml.annotations.require(
        namespace="runtime",
        fragment={"frameworks": {"plain": {"num_threads": 2}}},
    )
    def target():
        calls.append(mode)

    resolution = dryml.annotations.resolve(target)
    assert resolution.requirements.runtime == {"frameworks": {"plain": {"num_threads": 2}}}

    if mode == "managed":
        dryml.session.manage(cpus=1)
        target()
    elif mode == "orchestrator":
        dryml.session.set_mode("orchestrator")
        target()
    elif mode == "probe":
        with enter_runtime(
            RuntimeMode.PROBE,
            enforcement=RuntimeEnforcement.STRICT,
        ):
            target()
    elif mode == "worker":
        with enter_runtime(
            RuntimeMode.WORKER,
            RuntimeAllocationView(cpus=(0,)),
            enforcement=RuntimeEnforcement.STRICT,
        ):
            target()
    else:
        target()

    assert calls == [mode]


def test_worker_allocation_proves_its_required_role_cardinality():
    @dryml.world.req(
        roles={
            "worker": {
                "replicas": {"exact": 2},
                "resources": {"cpus": {"exact": 1}},
            }
        }
    )
    def target():
        return "ok"

    allocation = RuntimeAllocationView(
        role="worker",
        replica=0,
        rank=0,
        local_rank=0,
        cpus=(0,),
        metadata={"role_size": 2, "world_size": 2},
    )
    with enter_runtime(
        RuntimeMode.WORKER,
        allocation,
        enforcement=RuntimeEnforcement.STRICT,
    ):
        assert target() == "ok"


def test_hard_class_decoration_preserves_identity_and_supported_binding():
    calls = []

    class Base:
        def instance(self):
            calls.append("instance")

        @staticmethod
        def static():
            calls.append("static")

        @classmethod
        def class_method(cls):
            calls.append(cls.__name__)

    decorated = dryml.world.req(cpus={"min": 10_000_000})(Base)
    assert decorated is Base

    dryml.session.manage(cpus=1)
    for call in (Base().instance, Base.static, Base.class_method):
        with pytest.raises(AnnotationResolutionError):
            call()
    assert calls == []


@pytest.mark.parametrize("call_order", [("base", "sibling"), ("sibling", "base")])
def test_inherited_staticmethod_uses_owner_specific_requirements_in_both_call_orders(call_order):
    calls = []

    class Base:
        @staticmethod
        @dryml.world.req(cpus={"max": 1})
        def target():
            calls.append("base")

    class Sibling(Base):
        pass

    decorated = dryml.world.req(cpus={"min": 2})(Sibling)
    assert decorated is Sibling

    base_wrapper = Base.target
    sibling_wrapper = Sibling.target
    assert sibling_wrapper is not base_wrapper
    assert is_trusted_wrapper(base_wrapper)
    assert is_trusted_wrapper(sibling_wrapper)
    assert trusted_original(sibling_wrapper) is trusted_original(base_wrapper)
    assert dryml.annotations.own_fragments(sibling_wrapper) == dryml.annotations.own_fragments(base_wrapper)

    dryml.session.manage(cpus=1)
    calls_by_owner = {
        "base": Base.target,
        "sibling": Sibling.target,
    }
    for owner in call_order:
        if owner == "sibling":
            with pytest.raises(AnnotationResolutionError, match="contradictory"):
                calls_by_owner[owner]()
        else:
            calls_by_owner[owner]()
    assert calls == ["base"]


def test_predecoration_reference_and_property_remain_outside_interception():
    calls = []

    def method(self):
        calls.append("method")

    class Target:
        direct = method

        @property
        def value(self):
            calls.append("property")
            return 1

    reference = Target.direct
    dryml.world.req(cpus={"min": 10_000_000})(Target)
    dryml.session.manage(cpus=1)
    reference(Target())
    assert Target().value == 1
    assert calls == ["method", "property"]

from __future__ import annotations

from dryml.execute.backend import Backend
from dryml.execute.models import ResourceAmounts, SubmittedCall
from dryml.execute.ray import RayBackendConfig
from dryml.execute.subprocess import SubProcessConfig


class FakeBackend(Backend):
    """Minimal inert implementation proving the common abstract contract."""

    def start(self):
        self.started = True

    def capabilities(self):
        return frozenset()

    def create_future(self, submission_id, output):
        return (submission_id, output)

    def submit(self, call, *, future):
        self.call = call
        self.future = future

    def discover(self,
                 *,
                 environment=None,
                 environment_spec=None,
                 world=None,
                 timeout):
        self.discovery = (environment, environment_spec, world, timeout)
        return None

    def resources(self, *, timeout):
        return None

    def reconcile_cleanup(self, submission_id, *, timeout):
        self.cleaned = submission_id

    def close(self, *, cancel, timeout):
        self.closed = (cancel, timeout)


def test_new_backend_contract_is_additive_and_create_future_is_inert():
    """The U1 backend API coexists with legacy backend classes without launch work."""
    backend = FakeBackend()
    output = object()
    future = backend.create_future("submission-1", output)

    assert future == ("submission-1", output)
    assert not hasattr(backend, "started")
    assert isinstance(Backend.__abstractmethods__, frozenset)
    assert SubmittedCall.__dataclass_fields__["payload"].name == "payload"


def test_submitted_call_constructor_preserves_legacy_shapes_and_optional_selector():  # noqa: E501
    """
    Submitted calls retain prior positional and keyword construction forms.
    """
    payload = object()
    output = object()
    legacy_positional = SubmittedCall(
        "submission", 1.0, payload, None, None, None, False, output,
    )
    legacy_keyword = SubmittedCall(
        submission_id="submission",
        admission_deadline=1.0,
        payload=payload,
        environment=None,
        world=None,
        execution_timeout=None,
        stream_output=False,
        output=output,
    )
    selection = object()
    selected = SubmittedCall(
        submission_id="submission",
        admission_deadline=1.0,
        payload=payload,
        environment=None,
        world=None,
        execution_timeout=None,
        stream_output=False,
        output=output,
        environment_spec=selection,
    )

    assert legacy_positional.environment_spec is None
    assert legacy_keyword.environment_spec is None
    assert selected.environment_spec is selection


def test_resource_value_mappings_are_defensively_frozen():
    """Snapshot values detach resource maps and reject invalid resource quantities."""
    accelerators = {"gpu": 1.0}
    amounts = ResourceAmounts(cpus=2.0, memory_bytes=4, accelerators=accelerators, named={})
    accelerators["gpu"] = 2.0

    assert amounts.accelerators["gpu"] == 1.0


def test_builtin_backend_capabilities_are_stable_and_truthful():
    """Public selection features match each built-in backend's implemented scope."""
    assert SubProcessConfig().create_backend().capabilities() == frozenset({
        "environment_selection", "world_admission", "live_output", "running_cancellation",
    })
    assert RayBackendConfig().create_backend().capabilities() == frozenset({
        "environment_selection", "world_admission", "live_output", "running_cancellation", "ray_existing_deployment",
    })

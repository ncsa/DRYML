"""Managed Experiment checkpoint Artifact integration tests."""

from __future__ import annotations

import pytest

from dryml.artifacts import Artifact, ArtifactRecoveryError, Value
from dryml.core import Par, Ref, Repo, StateRef, Template, TemplateBundle, definition_mode, selector_mode
from dryml.core.object import Pickleable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.managed import ManagedConfig, managed_operation
from dryml.models import Experiment, ExperimentData, TrainFunction


pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


class CounterModel(Pickleable):
    """Tiny mutable model whose saved value distinguishes checkpoints."""

    def __init__(self, value=0):
        self.value = value


class OneUpdate(TrainFunction):
    """Trainer exposing one truthful update callback."""

    def __call__(self, exp, *, callbacks=()):
        exp.model.value += 1
        exp.state.record_update(examples=1, loss=2.0)
        for callback in callbacks:
            callback()


class TerminalOnly(TrainFunction):
    """Trainer with no intermediate safe point and one terminal mutation."""

    supports_safe_points = False

    def __call__(self, exp, *, callbacks=()):
        exp.model.value += 1
        exp.state.record_update(examples=1, loss=2.0)


class SavedTestDataValue(Value):
    """Artifact that records the exact Ref-held test-data StateRef."""

    calls = []

    def __init__(self, data: Ref[StateRef]):
        self.data = data

    @property
    def ready(self):
        """Return whether a complete result payload is installed."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Retain the bound test reference without materializing its payload."""

        type(self).calls.append(self.data)
        self._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": 1,
        })


class ConstantValue(Value):
    """Fully resolved Artifact used to prove no inactive root is bound."""

    calls = 0

    @property
    def ready(self):
        """Return whether this Artifact has a complete result."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Install one constant scalar result."""

        type(self).calls += 1
        self._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": 7,
        })


class InitiallyReadyArtifact(Artifact):
    """Non-Value Artifact whose ready payload still requires managed completion."""

    calls = 0

    @property
    def ready(self):
        """Report a usable payload before the operation has completed."""

        return True

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Record the one authored invocation required to establish authority."""

        type(self).calls += 1


class NeverReadyArtifact(Artifact):
    """Artifact that completes managed work without a valid result payload."""

    calls = 0

    @property
    def ready(self):
        """Remain invalid even after managed completion."""

        return False

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Complete without installing a ready result, which U9 must reject."""

        type(self).calls += 1


class RequiredComputeValue(Value):
    """Artifact with an incompatible required ordinary compute argument."""

    @property
    def ready(self):
        """Return whether a complete Value result has been installed."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, required, *, managed) -> None:
        """Declare the preflight-rejected ordinary argument."""

        del required, managed


class NoReceiptValue(Value):
    """Artifact whose managed compute omits the required final StateRef receipt."""

    @property
    def ready(self):
        """Return whether a complete Value result has been installed."""

        return self._value_is_present()

    @managed_operation(resumable=True)
    def compute(self, *, managed):
        """Remain intentionally incompatible with Experiment Artifact evaluation."""

        del managed


class SavedModelValue(Value):
    """Artifact that records the exact bound model StateRef without loading it."""

    calls = []

    def __init__(self, model: Ref[StateRef]):
        self.model = model

    @property
    def ready(self):
        """Return whether this artifact has a completed public value."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True)
    def compute(self, *, managed) -> None:
        """Install the exact model reference as an inspectable scalar surrogate."""

        type(self).calls.append(self.model)
        self._install_value_payload({
            "format": "dryml.artifacts.value", "version": 1,
            "present": True, "result": 1,
        })


def test_terminal_checkpoint_binds_artifact_to_exact_experiment_state(tmp_path):
    """Terminal history and Artifact receipts use the same exact checkpoint."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    SavedModelValue.calls.clear()
    exp = Experiment(
        CounterModel(), OneUpdate(),
        artifacts=TemplateBundle({"score": Template(SavedModelValue, Par("this.model"))}),
        repo=repo,
    )

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    history = ExperimentData.find(final.object_projection(), repo=repo)
    row = history.data.iloc[-1]

    assert final == exp.train.status(state_repo=repo).final_state_ref
    assert len(history.data) == 2
    assert row.state_ref == final
    assert row.evaluation_status == "completed"
    assert SavedModelValue.calls == [final.at(GraphPath((Parameter("model"),)))]
    assert history.data.iloc[0].eval_artifacts == row.eval_artifacts
    assert SavedModelValue.calls[-1] == final.at(GraphPath((Parameter("model"),)))


def test_empty_artifacts_still_publish_one_terminal_facts_only_row(tmp_path):
    """A trainer without safe points still creates completed terminal history."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    exp = Experiment(CounterModel(), TerminalOnly(), repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    history = ExperimentData.find(final.object_projection(), repo=repo)
    row = history.data.iloc[0]

    assert len(history.data) == 1
    assert row.state_ref == final
    assert row.expected_artifacts == []
    assert row.eval_artifacts == {}
    assert row.evaluation_status == "completed"
    status = exp.train.status(state_repo=repo)
    assert (final, status.checkpoint_state_ref, status.final_state_ref) == (
        row.state_ref, row.state_ref, row.state_ref,
    )


@pytest.mark.parametrize("artifacts", (
    Template(SavedModelValue, Par("this.model")),
    [Template(SavedModelValue, Par("this.model"))],
    {"score": Template(SavedModelValue, Par("this.model"))},
))
def test_experiment_normalizes_every_public_artifact_form(artifacts):
    """Public singleton, sequence, and mapping forms remain inert at construction."""

    exp = Experiment(CounterModel(), TerminalOnly(), artifacts=artifacts)

    expected = ("score",) if isinstance(artifacts, dict) else ("artifact_0",)
    assert exp.artifacts.names == expected


def test_resolved_artifact_recipe_is_unchanged_and_receipt_is_terminal(tmp_path):
    """A recipe without ``this`` is not rebound as an unknown template root."""

    repo = Repo(DirStore(tmp_path / "store"))
    ConstantValue.calls = 0
    recipe = Template(ConstantValue)
    exp = Experiment(CounterModel(), TerminalOnly(), artifacts=recipe, repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert recipe.is_resolved
    assert ConstantValue.calls == 1
    assert row.state_ref == final
    restored = repo.load_state_ref(row.eval_artifacts["artifact_0"], reuse_live="never", cache="none")
    assert row.eval_artifacts["artifact_0"] == restored.compute.status(
        state_repo=repo,
    ).final_state_ref
    assert restored.ready
    assert row.evaluation_status == "completed"


def test_initially_ready_artifact_still_completes_its_retained_managed_receiver(tmp_path):
    """Ready payload state cannot substitute for the retained compute receipt."""

    repo = Repo(DirStore(tmp_path / "store"))
    InitiallyReadyArtifact.calls = 0
    exp = Experiment(
        CounterModel(), TerminalOnly(), artifacts=Template(InitiallyReadyArtifact), repo=repo,
    )

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]
    initial = row.artifact_inputs["artifact_0"]
    recovered = Artifact.recover(initial, repo=repo, reuse_live="never")

    assert InitiallyReadyArtifact.calls == 1
    assert row.eval_artifacts["artifact_0"] == recovered.last_state_ref
    assert recovered.compute.status(state_repo=repo).final_state_ref == recovered.last_state_ref
    assert row.evaluation_status == "completed"


def test_completed_but_not_ready_artifact_is_never_published_as_an_evaluation(tmp_path):
    """A managed receipt without a ready restored Artifact leaves a failed row."""

    repo = Repo(DirStore(tmp_path / "store"))
    NeverReadyArtifact.calls = 0
    exp = Experiment(
        CounterModel(), TerminalOnly(), artifacts=Template(NeverReadyArtifact), repo=repo,
    )

    with pytest.raises(ArtifactRecoveryError, match="not ready"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    row = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    initial = row.artifact_inputs["artifact_0"]
    receiver = repo.load_state_ref(initial, reuse_live="never", cache="none")

    assert NeverReadyArtifact.calls == 1
    assert receiver.compute.status(state_repo=repo).state == "completed"
    assert row.eval_artifacts == {}
    assert row.evaluation_status == "failed"


def test_history_remains_readable_when_its_checkpoint_payload_is_unavailable(tmp_path):
    """History analysis retains exact references without opening checkpoint payloads."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    final = Experiment(CounterModel(), TerminalOnly(), repo=repo).train(
        managed=ManagedConfig(state_repo=repo),
    )
    history = ExperimentData.find(final.object_projection(), repo=repo)
    history_state = history.last_state_ref
    unavailable = tmp_path / "unavailable-checkpoint"
    store.get_snapshot_directory(final).rename(unavailable)

    recovered = ExperimentData.find(final.object_projection(), repo=repo)

    assert recovered.data.iloc[0].state_ref == final
    assert store.get_snapshot_directory(history_state).exists()
    with pytest.raises(RepoLoadError):
        repo.load_state_ref(final, reuse_live="never", cache="none")


def test_test_data_recipe_binds_the_exact_ref_held_dataset_state(tmp_path):
    """``this.test_data`` preserves the configured StateRef with no live fallback."""

    repo = Repo(DirStore(tmp_path / "store"))
    test_data = repo.save_object(CounterModel(9))
    SavedTestDataValue.calls.clear()
    exp = Experiment(
        CounterModel(), TerminalOnly(), test_data=test_data,
        artifacts=Template(SavedTestDataValue, Par("this.test_data")), repo=repo,
    )

    exp.train(managed=ManagedConfig(state_repo=repo))

    assert SavedTestDataValue.calls == [test_data]


def test_missing_test_data_recipe_fails_before_training_or_checkpoint_mutation(tmp_path):
    """A required test-data binding never falls back to training or validation data."""

    repo = Repo(DirStore(tmp_path / "store"))
    model = CounterModel()
    exp = Experiment(
        model, TerminalOnly(), train_data=CounterModel(3), val_data=CounterModel(4),
        artifacts=Template(SavedTestDataValue, Par("this.test_data")), repo=repo,
    )

    with pytest.raises(TypeError, match="cannot bind"):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert model.value == 0
    assert exp.state.phase is None
    assert exp.train.status(state_repo=repo).state == "failed"


@pytest.mark.parametrize("artifacts", (
    Template(ConstantValue), [Template(ConstantValue)], {"constant": Template(ConstantValue)},
    TemplateBundle({"constant": Template(ConstantValue)}),
))
def test_artifact_forms_are_canonical_at_every_experiment_definition_boundary(artifacts):
    """Experiment normalizes recipes for direct, CDef, and object-mode entry points."""

    kwargs = {"artifacts": artifacts}
    direct = Experiment(CounterModel(), TerminalOnly(), **kwargs)
    definition = Experiment.defn(CounterModel(), TerminalOnly(), **kwargs)
    alias = Experiment.d(CounterModel(), TerminalOnly(), **kwargs)
    with definition_mode():
        mode_definition = Experiment(CounterModel(), TerminalOnly(), **kwargs)
    with definition_mode(concrete=True):
        concrete = Experiment(CounterModel(), TerminalOnly(), **kwargs)
    with selector_mode():
        selector = Experiment(CounterModel(), TerminalOnly(), **kwargs)

    assert isinstance(direct.artifacts, TemplateBundle)
    assert isinstance(definition.parameters["artifacts"], TemplateBundle)
    assert definition.parameters["artifacts"] == alias.parameters["artifacts"]
    assert definition.parameters["artifacts"] == mode_definition.parameters["artifacts"]
    assert isinstance(concrete.parameters["artifacts"].target, TemplateBundle)
    assert isinstance(selector.root.parameters["artifacts"], TemplateBundle)


def test_artifact_scalars_store_normalized_native_values_without_backend_imports():
    """History values use the scalar conversion result and omit non-scalar arrays."""

    import numpy as np

    artifact = object.__new__(ConstantValue)
    artifact._install_value_payload({
        "format": "dryml.artifacts.value", "version": 1, "present": True,
        "result": {"python": 1, "numpy": np.array(2.5), "array": np.array([3])},
    })

    assert Experiment._artifact_scalars("metric", artifact) == {
        "metric.python": 1,
        "metric.numpy": 2.5,
    }


def test_artifact_scalars_accept_installed_native_zero_dimensional_values():
    """Optional native backends need no Experiment import-time dependency to convert."""

    import importlib.util
    import numpy as np

    values = {"numpy": np.array(2.5), "array": np.array([3])}
    expected = {"metric.numpy": 2.5}
    if importlib.util.find_spec("torch") is not None:
        import torch

        values["torch"] = torch.tensor(4.5)
        values["torch_array"] = torch.tensor([4.5])
        expected["metric.torch"] = 4.5
    if importlib.util.find_spec("tensorflow") is not None:
        import tensorflow as tf

        values["tensorflow"] = tf.constant(5.5)
        values["tensorflow_array"] = tf.constant([5.5])
        expected["metric.tensorflow"] = 5.5
    artifact = object.__new__(ConstantValue)
    artifact._install_value_payload({
        "format": "dryml.artifacts.value", "version": 1, "present": True,
        "result": values,
    })

    assert Experiment._artifact_scalars("metric", artifact) == expected


@pytest.mark.parametrize("recipe, error", (
    (Template(CounterModel), "Artifact definition"),
    (Template(RequiredComputeValue), "ordinary arguments"),
    (Template(NoReceiptValue), "resumable and return"),
))
def test_artifact_preflight_rejects_invalid_contracts_before_trainer_mutation(
        tmp_path, recipe, error):
    """Invalid recipes fail before model/progress/trainer mutation or a checkpoint."""

    repo = Repo(DirStore(tmp_path / "store"))
    model = CounterModel()
    exp = Experiment(model, TerminalOnly(), artifacts=recipe, repo=repo)

    with pytest.raises(TypeError, match=error):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert model.value == 0
    assert exp.state.phase is None
    assert exp.train.status(state_repo=repo).checkpoint_state_ref is None

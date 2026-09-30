"""Managed Experiment checkpoint Artifact integration tests."""

from __future__ import annotations

import pytest

from dryml.artifacts import Artifact, ArtifactRecoveryError, Value
from dryml.core import Definition, Par, Ref, Repo, StateRef, definition_mode, selector_mode
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
        artifacts={"score": Definition(SavedModelValue, Par("this.model"))},
        repo=repo, checkpoint_every_steps=1,
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


def test_symbolic_metric_recipe_binds_through_ref_during_evaluation(
        tmp_path, monkeypatch):
    """Preflight and runtime bind metric roots hidden by Fold's source Ref."""

    from dryml.metrics import regressor_mae

    repo = Repo(DirStore(tmp_path / "store"))
    test_data = repo.save_object(CounterModel())
    recipe = regressor_mae(
        Par("this.test_data"), Par("this.model"), mode="global"
    )
    exp = Experiment(
        CounterModel(), TerminalOnly(), test_data=test_data,
        artifacts={"score": recipe}, repo=repo,
    )

    assert recipe.names == ()
    exp._preflight_artifacts()

    observed = []

    def inspect_bound(definition, *, repo=None):
        from dryml.core.template import _snapshot_parameters

        del repo
        observed.append(tuple(_snapshot_parameters(definition, traverse_refs=True)))
        raise RuntimeError("stop after runtime binding")

    monkeypatch.setattr(Definition, "concretize", inspect_bound)
    checkpoint = StateRef(
        exp.object_ref,
        {path: "dryml-" + "0" * 64 for path in exp.object_ref.objects},
    )
    with pytest.raises(RuntimeError, match="stop after runtime binding"):
        exp._evaluate_artifact(
            "score", recipe, checkpoint, None, None,
            type("Context", (), {"state_repo": repo})(),
        )

    assert observed == [()]


class FiveUpdates(TrainFunction):
    """Trainer exposing five post-update safe points for cadence tests."""

    def __call__(self, exp, *, callbacks=()):
        for _ in range(5):
            exp.model.value += 1
            exp.state.record_update(examples=1, loss=1.0)
            for callback in callbacks:
                callback()


@pytest.mark.parametrize("cadence, expected_rows", ((None, 1), (2, 3)))
def test_experiment_checkpoint_cadence_filters_intermediates_but_keeps_terminal(
        tmp_path, cadence, expected_rows):
    """Only exact retained-step multiples checkpoint before the mandatory final row."""

    repo = Repo(DirStore(tmp_path / "store"))
    exp = Experiment(CounterModel(), FiveUpdates(), checkpoint_every_steps=cadence, repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    history = ExperimentData.find(final.object_projection(), repo=repo).data

    assert len(history) == expected_rows
    assert list(history.examples_seen) == ([5] if cadence is None else [2, 4, 5])
    assert exp.state.step == exp.model.value == 5


@pytest.mark.parametrize("cadence", (0, -1, True, 1.5))
def test_experiment_rejects_invalid_checkpoint_cadence_before_state_mutation(cadence):
    """Cadence validation is exact and leaves model/training state untouched."""

    model = CounterModel()
    with pytest.raises((TypeError, ValueError)):
        Experiment(model, TerminalOnly(), checkpoint_every_steps=cadence)
    assert model.value == 0


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
    Definition(SavedModelValue, Par("this.model")),
    [Definition(SavedModelValue, Par("this.model"))],
    {"score": {"nested": Definition(SavedModelValue, Par("this.model"))}},
))
def test_experiment_rejects_retired_artifact_conveniences_before_training(artifacts):
    """Only the direct flat named mapping reaches Experiment construction."""

    model = CounterModel()
    with pytest.raises(Exception):
        Experiment(model, TerminalOnly(), artifacts=artifacts)
    assert model.value == 0


def test_resolved_artifact_recipe_is_unchanged_and_receipt_is_terminal(tmp_path):
    """A recipe without ``this`` is not rebound as an unknown template root."""

    repo = Repo(DirStore(tmp_path / "store"))
    ConstantValue.calls = 0
    recipe = Definition(ConstantValue)
    exp = Experiment(CounterModel(), TerminalOnly(), artifacts={"constant": recipe}, repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert recipe.is_resolved
    assert ConstantValue.calls == 1
    assert row.state_ref == final
    restored = repo.load_state_ref(row.eval_artifacts["constant"], reuse_live="never", cache="none")
    assert row.eval_artifacts["constant"] == restored.compute.status(
        state_repo=repo,
    ).final_state_ref
    assert restored.ready
    assert row.evaluation_status == "completed"


def test_initially_ready_artifact_still_completes_its_retained_managed_receiver(tmp_path):
    """Ready payload state cannot substitute for the retained compute receipt."""

    repo = Repo(DirStore(tmp_path / "store"))
    InitiallyReadyArtifact.calls = 0
    exp = Experiment(
        CounterModel(), TerminalOnly(), artifacts={"initial": Definition(InitiallyReadyArtifact)}, repo=repo,
    )

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]
    initial = row.artifact_inputs["initial"]
    recovered = Artifact.recover(initial, repo=repo, reuse_live="never")

    assert InitiallyReadyArtifact.calls == 1
    assert row.eval_artifacts["initial"] == recovered.last_state_ref
    assert recovered.compute.status(state_repo=repo).final_state_ref == recovered.last_state_ref
    assert row.evaluation_status == "completed"


def test_completed_but_not_ready_artifact_is_never_published_as_an_evaluation(tmp_path):
    """A managed receipt without a ready restored Artifact leaves a failed row."""

    repo = Repo(DirStore(tmp_path / "store"))
    NeverReadyArtifact.calls = 0
    exp = Experiment(
        CounterModel(), TerminalOnly(), artifacts={"never": Definition(NeverReadyArtifact)}, repo=repo,
    )

    with pytest.raises(ArtifactRecoveryError, match="not ready"):
        exp.train(managed=ManagedConfig(state_repo=repo))
    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    row = ExperimentData.find(checkpoint.object_projection(), repo=repo).data.iloc[0]
    initial = row.artifact_inputs["never"]
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
        artifacts={"test-data": Definition(SavedTestDataValue, Par("this.test_data"))}, repo=repo,
    )

    exp.train(managed=ManagedConfig(state_repo=repo))

    assert SavedTestDataValue.calls == [test_data]


def test_missing_test_data_recipe_fails_before_training_or_checkpoint_mutation(tmp_path):
    """A required test-data binding never falls back to training or validation data."""

    repo = Repo(DirStore(tmp_path / "store"))
    model = CounterModel()
    exp = Experiment(
        model, TerminalOnly(), train_data=CounterModel(3), val_data=CounterModel(4),
        artifacts={"test-data": Definition(SavedTestDataValue, Par("this.test_data"))}, repo=repo,
    )

    with pytest.raises(TypeError, match="cannot bind"):
        exp.train(managed=ManagedConfig(state_repo=repo))

    assert model.value == 0
    assert exp.state.phase is None
    assert exp.train.status(state_repo=repo).state == "failed"


def test_artifact_mapping_is_inert_until_activated_concretization():
    """Definition and selector authoring retain frozen mapping data until activation."""

    artifacts = {"constant": Definition(ConstantValue)}
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

    assert tuple(direct.artifacts) == ("constant",)
    assert tuple(definition.parameters["artifacts"]) == ("constant",)
    assert definition.parameters["artifacts"] == alias.parameters["artifacts"]
    assert definition.parameters["artifacts"] == mode_definition.parameters["artifacts"]
    assert tuple(concrete.parameters["artifacts"]) == ("constant",)
    assert tuple(selector.root.parameters["artifacts"]) == ("constant",)


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
    ({"invalid": Definition(CounterModel)}, "Artifact definition"),
    ({"required": Definition(RequiredComputeValue)}, "ordinary arguments"),
    ({"no-receipt": Definition(NoReceiptValue)}, "resumable and return"),
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

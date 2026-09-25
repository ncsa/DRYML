"""Streaming regression and single-label classification metric coverage."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.data import Dataset, Map, Pipe, Project, Select
from dryml.managed import ManagedConfig
from dryml.methods import AccumulatorGroup, Method
from dryml.models import Model


pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def test_confusion_inference_constructs_no_helper_methods(monkeypatch):
    """Pure transition inference must not construct an initializer Method."""
    from dryml.metrics import reductions

    observation = {
        "prediction": TensorSpec("int64", shape=(), backend="numpy"),
        "target": TensorSpec("int64", shape=(), backend="numpy"),
    }
    transition = reductions.ConfusionCounts((0, 1))
    expected = reductions.ConfusionInitial((0, 1)).infer_output_spec(observation)

    def forbid_construction(*args, **kwargs):
        raise AssertionError("inference constructed a helper Method")

    monkeypatch.setattr(reductions, "ConfusionInitial", forbid_construction)
    assert transition.infer_output_spec(observation, expected) == expected


class EvaluationDataset(Dataset):
    """Small re-iterable batched source that records evaluations by class."""

    iterations = 0

    def __init__(self, batches, *, dtype="float64"):
        self.batches = tuple({key: np.asarray(value) for key, value in batch.items()} for batch in batches)
        super().__init__(spec={
            "x": TensorSpec(dtype, shape=(), batch=Dynamic, backend="numpy"),
            "y": TensorSpec(dtype, shape=(), batch=Dynamic, backend="numpy"),
        })

    def __iter__(self):
        """Yield each declared uneven batch and record one source traversal."""

        type(self).iterations += 1
        yield from self.batches


class IdentityModel(Model):
    """Model Method that returns its input while recording native predictions."""

    calls = 0

    def __init__(self, *, dtype="float64", backend="numpy"):
        super().__init__(output_spec=TensorSpec(dtype, shape=(), backend=backend))

    def __call__(self, value):
        """Return the already-decoded input label or scalar prediction."""

        type(self).calls += 1
        return value


class IdentityLabels(Method):
    """Explicit caller-owned label conversion that preserves decoded labels."""

    def __call__(self, value):
        """Return one caller-decoded label tensor unchanged."""

        return value

    def infer_output_spec(self, input_spec):
        """Preserve one decoded label specification without execution."""

        return input_spec


class NativeEvaluationDataset(Dataset):
    """Construct backend-native numeric batches only when a Fold iterates it."""

    iterations = 0

    def __init__(self, backend, batches, *, dtype):
        self.backend = backend
        self.batches = tuple(batches)
        self.dtype = dtype
        super().__init__(spec={
            "x": TensorSpec(dtype, shape=(), batch=Dynamic, backend=backend),
            "y": TensorSpec(dtype, shape=(), batch=Dynamic, backend=backend),
        })

    def __iter__(self):
        """Yield fresh native arrays while counting one declared source traversal."""

        type(self).iterations += 1
        for x, y in self.batches:
            yield {
                "x": _native_tensor(self.backend, x, self.dtype),
                "y": _native_tensor(self.backend, y, self.dtype),
            }


class SnapshotOffsetModel(Pickleable, Model):
    """Serializable model whose saved offset distinguishes exact selected states."""

    def __init__(self, offset=0.0):
        Model.__init__(self, output_spec=TensorSpec("float64", shape=(), backend="numpy"))
        self.offset = offset

    def __call__(self, value):
        """Return a native prediction offset by the persisted model state."""

        return value + self.offset


def _native_tensor(backend, values, dtype):
    """Create one CPU tensor for the requested optional numerical backend."""

    if backend == "numpy":
        return np.asarray(values, dtype=dtype)
    if backend == "torch":
        import dryml.torch
        import torch

        return torch.tensor(values, dtype=getattr(torch, dtype))
    import dryml.tf
    import tensorflow as tf

    return tf.constant(values, dtype=getattr(tf, dtype))


def _host_values(value):
    """Normalize an asserted terminal native result after its Fold has completed."""

    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    if type(value).__module__.startswith("tensorflow"):
        return value.numpy()
    return value


def test_confusion_program_counts_truth_rows_and_prediction_columns():
    """The declared confusion initializer and transition retain a fixed class domain."""

    from dryml.metrics import ConfusionCounts, ConfusionInitial

    initial = ConfusionInitial(classes=(0, 1))
    counts = ConfusionCounts(classes=(0, 1))
    state = initial({"prediction": np.array([0, 1]), "target": np.array([0, 1])})

    actual = counts(
        {"prediction": np.array([1, 0, 1]), "target": np.array([0, 1, 1])},
        state,
    )

    assert np.array_equal(actual, np.array([[0, 1], [1, 1]], dtype=np.int64))


def test_confusion_result_methods_define_all_f1_modes_and_zero_policies():
    """Counts derive accuracy/F1 natively, including unsupported class terms."""

    from dryml.metrics import AccuracyFromConfusion, F1FromConfusion

    matrix = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64)

    assert AccuracyFromConfusion()(matrix) == pytest.approx(0.75)
    assert np.allclose(F1FromConfusion(average="none")(matrix), [0.8, 2 / 3, 0.0])
    assert F1FromConfusion(average="micro")(matrix) == pytest.approx(0.75)
    assert F1FromConfusion(average="macro")(matrix) == pytest.approx((0.8 + 2 / 3) / 3)
    assert F1FromConfusion(average="weighted")(matrix) == pytest.approx((0.8 * 3 + (2 / 3)) / 4)
    assert F1FromConfusion(average="binary", positive_index=1)(np.array([[1, 1], [0, 1]], dtype=np.int64)) == pytest.approx(2 / 3)
    assert AccuracyFromConfusion()(np.zeros((2, 2), dtype=np.int64)) == 0.0
    assert F1FromConfusion(average="weighted")(np.zeros((2, 2), dtype=np.int64)) == 0.0
    assert np.array_equal(F1FromConfusion(average="none")(np.zeros((2, 2), dtype=np.int64)), [0.0, 0.0])


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_confusion_counts_preserve_native_integer_carry_on_required_backends(backend):
    """NumPy, Torch CPU, and TensorFlow CPU count decoded integer labels natively."""

    from dryml.metrics import AccuracyFromConfusion, ConfusionCounts, ConfusionInitial, F1FromConfusion

    if backend == "numpy":
        labels = lambda values: np.asarray(values, dtype=np.int64)
    elif backend == "torch":
        import dryml.torch
        import torch
        labels = lambda values: torch.tensor(values, dtype=torch.int64)
    else:
        import dryml.tf
        import tensorflow as tf
        labels = lambda values: tf.constant(values, dtype=tf.int64)
    observation = {"prediction": labels([1, 0, 1]), "target": labels([0, 1, 1])}
    state = ConfusionInitial((0, 1))(observation)
    actual = ConfusionCounts((0, 1))(observation, state)

    assert type(actual).__module__.startswith(backend if backend != "tf" else "tensorflow")
    host = actual if backend == "numpy" else (actual.detach().cpu().numpy() if backend == "torch" else actual.numpy())
    assert np.array_equal(host, [[0, 1], [1, 1]])
    assert _host_values(AccuracyFromConfusion()(actual)) == pytest.approx(1 / 3)
    assert _host_values(F1FromConfusion(average="macro")(actual)) == pytest.approx(0.25)


def test_confusion_rejects_invalid_domains_labels_and_overflow_before_mutation():
    """Domain, shape, unknown-label, and overflow errors keep the input carry intact."""

    from dryml.data.reduction_methods import _MAX_INT64
    from dryml.metrics import AccuracyFromConfusion, ConfusionCounts, ConfusionInitial, F1FromConfusion

    for classes in ((), (0, 0), (0, "one"), (True, False)):
        with pytest.raises((TypeError, ValueError)):
            ConfusionCounts(classes)
    counts = ConfusionCounts((0, 1))
    state = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        counts({"prediction": np.array([1]), "target": np.array([2])}, state)
    assert not state.any()
    with pytest.raises(ValueError, match="one-dimensional"):
        counts({"prediction": np.array([[1, 0]]), "target": np.array([[1, 0]])}, state)
    with pytest.raises(ValueError, match="matching backend and shape"):
        counts({"prediction": np.array([1, 0]), "target": np.array([1])}, state)
    state[0, 0] = _MAX_INT64
    with pytest.raises(OverflowError, match="overflow"):
        counts({"prediction": np.array(0), "target": np.array(0)}, state)
    assert state[0, 0] == _MAX_INT64
    with pytest.raises(ValueError, match="positive_index"):
        from dryml.metrics import F1FromConfusion
        F1FromConfusion(average="binary")
    with pytest.raises(ValueError, match="positive_index"):
        from dryml.metrics import F1FromConfusion
        F1FromConfusion(average="macro", positive_index=0)
    with pytest.raises(ValueError, match="exactly two"):
        F1FromConfusion(average="binary", positive_index=0)(np.ones((3, 3), dtype=np.int64))
    string_observation = {"prediction": np.array(["dog", "cat"]), "target": np.array(["cat", "dog"])}
    string_state = ConfusionInitial(("cat", "dog"))(string_observation)
    assert np.array_equal(
        ConfusionCounts(("cat", "dog"))(string_observation, string_state),
        [[0, 1], [1, 0]],
    )
    for matrix in (
        np.ones((2, 3), dtype=np.int64),
        np.array([[1, -1], [0, 1]], dtype=np.int64),
        np.ones((2, 2), dtype=np.float64),
    ):
        with pytest.raises((TypeError, ValueError)):
            AccuracyFromConfusion()(matrix)
        with pytest.raises((TypeError, ValueError)):
            F1FromConfusion(average="macro")(matrix)


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_confusion_methods_keep_native_values_until_the_terminal_boundary(monkeypatch, backend):
    """Native count transitions and result Methods avoid host tensor extraction."""

    from dryml.metrics import AccuracyFromConfusion, ConfusionCounts, ConfusionInitial, F1FromConfusion

    labels = _native_tensor(backend, (1, 0, 1), "int64")
    target = _native_tensor(backend, (0, 1, 1), "int64")
    observation = {"prediction": labels, "target": target}
    monkeypatch.setattr(
        np,
        "asarray",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected np.asarray")),
    )
    if backend == "torch":
        import torch

        for name in ("cpu", "item", "numpy"):
            monkeypatch.setattr(
                torch.Tensor,
                name,
                lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError(f"unexpected Tensor.{name}")),
            )
    else:
        original_numpy = type(labels).numpy

        def guard_direct_numpy(value, *_args, **_kwargs):
            """Allow TensorFlow internals but reject direct metrics implementation extraction."""

            import inspect

            caller = inspect.currentframe().f_back
            if caller is not None and caller.f_code.co_filename.endswith("metrics/reductions.py"):
                raise AssertionError("unexpected direct Tensor.numpy")
            return original_numpy(value, *_args, **_kwargs)

        monkeypatch.setattr(type(labels), "numpy", guard_direct_numpy)

    state = ConfusionInitial((0, 1))(observation)
    counts = ConfusionCounts((0, 1))(observation, state)
    accuracy = AccuracyFromConfusion()(counts)
    f1 = F1FromConfusion(average="macro")(counts)

    monkeypatch.undo()
    assert _host_values(accuracy) == pytest.approx(1 / 3)
    assert _host_values(f1) == pytest.approx(0.25)


def test_regression_factories_are_inert_cdef_evaluations_and_weight_uneven_batches(tmp_path):
    """MAE/MSE use one declared Diff/Abs-or-Squared stream and U6 mean semantics."""

    from dryml.artifacts import Fold
    from dryml.core import ConcreteDefinition, Repo
    from dryml.metrics import regressor_mae, regressor_mse

    EvaluationDataset.iterations = IdentityModel.calls = 0
    source = EvaluationDataset((
        {"x": [1.0, 5.0], "y": [2.0, 2.0]},
    ))
    model = IdentityModel()
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source_ref, model_ref = repo.save_object(source), repo.save_object(model)
    mae = regressor_mae(source_ref, model_ref, mode="global")
    mse = regressor_mse(source_ref, model_ref, mode="global")

    assert isinstance(mae, Fold)
    assert isinstance(mae.src, ConcreteDefinition)
    from dryml.core.cdef_graph import ConcreteDefinitionGraph

    helper_names = {
        node.definition.cls.qualname
        for node in ConcreteDefinitionGraph.from_root(mae.definition, expand_ref_targets=True).nodes()
    }
    assert {"Diff", "Abs", "MeanInitial", "MeanUpdate", "MeanFinalize"} <= helper_names
    assert not mae.ready and not mse.ready
    assert (EvaluationDataset.iterations, IdentityModel.calls) == (0, 0)
    repo.save_object(mae)
    repo.save_object(mse)
    store.commit()
    mae.compute(managed=ManagedConfig(state_repo=repo))
    mse.compute(managed=ManagedConfig(state_repo=repo))

    assert mae.value() == pytest.approx(2.0)
    assert mse.value() == pytest.approx(5.0)


def test_classification_factories_use_explicit_labels_and_one_traversal_each(tmp_path):
    """Each factory folds one explicit evaluation stream without implicit decoding."""

    from dryml.core import Repo
    from dryml.metrics import classifier_accuracy, classifier_confusion_matrix, classifier_f1

    EvaluationDataset.iterations = IdentityModel.calls = 0
    source = EvaluationDataset((
        {"x": [0, 1], "y": [0, 1]},
        {"x": [1], "y": [0]},
    ), dtype="int64")
    model = IdentityModel(dtype="int64")
    labels = IdentityLabels()
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    source_ref, model_ref = repo.save_object(source), repo.save_object(model)
    confusion = classifier_confusion_matrix(source_ref, model_ref, classes=(0, 1), prediction_labels=labels, target_labels=labels)
    accuracy = classifier_accuracy(source_ref, model_ref, classes=(0, 1), prediction_labels=labels, target_labels=labels)
    f1 = classifier_f1(source_ref, model_ref, classes=(0, 1), prediction_labels=labels, target_labels=labels, average="macro")

    for name, fold in (("confusion", confusion), ("accuracy", accuracy), ("f1", f1)):
        repo.save_object(fold)
        store.commit()
        fold.compute(managed=ManagedConfig(state_repo=repo))
    assert np.array_equal(confusion.value(), [[1, 1], [0, 1]])
    assert accuracy.value() == pytest.approx(2 / 3)
    assert f1.value() == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert EvaluationDataset.iterations == 3
    assert IdentityModel.calls == 6


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_regression_factories_keep_native_batches_until_terminal_result(tmp_path, backend):
    """MAE/MSE retain native batch arithmetic and the U6 uneven-batch denominator."""

    from dryml.metrics import regressor_mae, regressor_mse

    NativeEvaluationDataset.iterations = IdentityModel.calls = 0
    source = NativeEvaluationDataset(
        backend,
        (((1.0, 5.0), (2.0, 2.0)), ((3.0,), (3.0,))),
        dtype="float64",
    )
    model = IdentityModel(dtype="float64", backend=backend)
    mae = regressor_mae(source, model, mode="global")
    mse = regressor_mse(source, model, mode="global")

    assert not mae.ready and not mse.ready
    assert (NativeEvaluationDataset.iterations, IdentityModel.calls) == (0, 0)
    assert source.last_state_ref is None and model.last_state_ref is None
    mae.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "mae")))
    mse.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "mse")))

    assert mae.value() == pytest.approx(4 / 3)
    assert mse.value() == pytest.approx(10 / 3)
    assert NativeEvaluationDataset.iterations == 2
    assert IdentityModel.calls == 4


def test_declared_confusion_group_produces_all_results_from_one_stream(tmp_path):
    """One declared Fold group emits confusion, accuracy, and F1 after one traversal."""

    from dryml.artifacts import Fold
    from dryml.metrics import AccuracyFromConfusion, ConfusionCounts, ConfusionInitial, F1FromConfusion

    EvaluationDataset.iterations = IdentityModel.calls = 0
    source = EvaluationDataset((
        {"x": [0, 1], "y": [0, 1]},
        {"x": [1], "y": [0]},
    ), dtype="int64")
    model = IdentityModel(dtype="int64")
    evaluation = Map.defn(
        source,
        Project.defn(
            prediction=Pipe.defn(Select.defn("x"), model),
            target=Select.defn("y"),
        ),
    ).concretize()
    fold = Fold(
        evaluation,
        initial_state=Project(
            ConfusionInitial((0, 1)),
            ConfusionInitial((0, 1)),
            ConfusionInitial((0, 1)),
        ),
        accumulator=AccumulatorGroup((
            ConfusionCounts((0, 1)),
            ConfusionCounts((0, 1)),
            ConfusionCounts((0, 1)),
        )),
        finalize=Project(
            confusion=Select(0),
            accuracy=Pipe(Select(1), AccuracyFromConfusion()),
            f1=Pipe(Select(2), F1FromConfusion(average="macro")),
        ),
    )

    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "store")))

    assert np.array_equal(fold.value()["confusion"], [[1, 1], [0, 1]])
    assert fold.value()["accuracy"] == pytest.approx(2 / 3)
    assert fold.value()["f1"] == pytest.approx(2 / 3)
    assert (EvaluationDataset.iterations, IdentityModel.calls) == (1, 2)


def test_factory_preserves_selected_model_state_and_restores_result_without_inputs(tmp_path):
    """A factory retains its nested exact model StateRef and result-only restoration stays inert."""

    from dryml.core import Repo
    from dryml.core.repo import RepoLoadError
    from dryml.metrics import regressor_mse

    input_store = DirStore(tmp_path / "inputs")
    input_repo = Repo(input_store)
    source = EvaluationDataset(({"x": [1.0, 5.0], "y": [2.0, 2.0]},))
    model = SnapshotOffsetModel(repo=input_repo)
    source_ref = input_repo.save_object(source)
    selected_model_ref = input_repo.save_object(model)
    model.offset = 10.0
    changed_model_ref = input_repo.save_object(model)
    fold = regressor_mse(source_ref, selected_model_ref, mode="global")

    assert selected_model_ref != changed_model_ref
    assert not fold.ready
    input_repo.save_object(fold)
    fold.compute(managed=ManagedConfig(state_repo=input_repo))
    assert fold.value() == pytest.approx(5.0)

    result_store = DirStore(tmp_path / "results")
    result_state = Repo(result_store).save_object(fold)
    restored = Repo(result_store).load_state_ref(result_state, reuse_live="never")

    assert restored.ready
    assert restored.value() == pytest.approx(5.0)
    with pytest.raises(RepoLoadError):
        restored.compute(managed=ManagedConfig(state_repo=Repo(result_store), rerun=True))

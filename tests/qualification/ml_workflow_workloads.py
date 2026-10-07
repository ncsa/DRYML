"""Real opt-in W1/W2/W3 builders and closed qualification evidence contracts.

Importing this module is NumPy-only. TensorFlow, Torch, JAX, Flax, Optax, TFDS,
pandas, Stores, and training are reached only by a selected real runner after
manifest preflight.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np

from dryml.artifacts import Value
from dryml.artifacts.value import _VALUE_FORMAT, _VALUE_VERSION
from dryml.core import AutoRef, ConcreteDefinition, Definition, ObjectRef, Ref, StateRef
from dryml.data import ArgMax, ArrayDataset, Batch, Cast, Dataset, Flatten, Map, Pipe, Project, Select, as_supervised
from dryml.data.image import ImageNormalize
from dryml.managed import ManagedConfig, ManagedContext, managed_operation

from .ml_workflow_fixtures import (
    FixtureManifest, FixtureManifestError, QualificationUnrun, REQUIRED_ENVIRONMENT_KEYS, _reference_from_json,
    _reference_to_json, config_digest, preflight_manifest,
)


THRESHOLDS = {"W1": 0.80, "W2": 0.12, "W3": 0.05}
WORKLOADS = ("W1", "W2", "W3")
FRAMEWORKS = ("tf", "torch", "jax")
EXECUTION_MODES = ("local", "managed-local", "subprocess", "ray")
CASE_KINDS = ("matrix", "tfds-tensorflow-to-torch")
_METRIC_NAMES = {"W1": "accuracy", "W2": "reconstruction_mse", "W3": "test_mse"}
_FORMULAS = {"W1": "categorical_accuracy", "W2": "normalized_pixel_mse", "W3": "noisy_observation_mse"}
_EXECUTION_BACKENDS = {
    "local": "core-local", "managed-local": "core-local",
    "subprocess": "core-subprocess", "ray": "core-ray",
}


class _QualificationMetric(Value[float]):
    """Compute one qualification metric from exact Dataset and Model authority."""

    def __init__(
            self, test_ds: Ref[AutoRef], model: Ref[AutoRef], *,
            metric: str, x: object, y: object, fixture_store: str | None,
    ) -> None:
        if metric not in {"accuracy", "mse"}:
            raise ValueError("Qualification metric must be accuracy or mse.")
        if fixture_store is not None and (type(fixture_store) is not str or not fixture_store):
            raise ValueError("Qualification metric fixture Store must be a nonempty path or None.")
        self.test_ds = test_ds
        self.model = model
        self.metric = metric
        self.x = x
        self.y = y
        self.fixture_store = fixture_store

    @property
    def ready(self) -> bool:
        """Return whether the complete scalar result is installed."""

        return self._value_is_present()

    @managed_operation(resumable=True, return_state_ref=True, store_parameter="store")
    def compute(self, *, store=None, managed: ManagedContext) -> None:
        """Evaluate the exact saved model over the exact saved test Dataset."""

        del store
        from dryml.core import Repo
        from dryml.core.store.dir import DirStore

        fixture_repo = None
        cursor = None
        try:
            if self.fixture_store is None:
                source = self._load(self.test_ds, managed.state_repo)
            else:
                fixture_repo = Repo(DirStore.open_existing(
                    self.fixture_store,
                    query_index="none",
                ))
                source = self._load(self.test_ds, fixture_repo)
            model = self._load(self.model, managed.state_repo)
            if not isinstance(source, Dataset) or not callable(model):
                raise TypeError("Qualification metric requires Dataset and callable Model authority.")
            numerator = 0.0
            denominator = 0
            cursor = source.iterator()
            for sample in cursor:
                target = self._array(self._select(sample, self.y))
                prediction = self._array(model(self._select(sample, self.x)))
                if self.metric == "accuracy":
                    prediction = np.argmax(prediction, axis=-1)
                    matches = np.asarray(prediction == target)
                    numerator += float(np.sum(matches))
                    denominator += int(matches.size)
                else:
                    difference = prediction - target
                    numerator += float(np.sum(difference * difference))
                    denominator += int(difference.size)
        finally:
            if cursor is not None:
                cursor.close()
            if fixture_repo is not None:
                fixture_repo.close(flush=False)
        if denominator == 0:
            raise ValueError("Qualification metric cannot evaluate an empty Dataset.")
        result = numerator / denominator
        if not math.isfinite(result):
            raise ValueError("Qualification metric result must be finite.")
        self._install_value_payload({
            "format": _VALUE_FORMAT,
            "version": _VALUE_VERSION,
            "present": True,
            "result": result,
        })

    @staticmethod
    def _load(reference, repo):
        """Materialize one retained reference through the selected state Repo."""

        if isinstance(reference, StateRef):
            return repo.load_state_ref(
                reference,
                reuse_live="never",
                cache="none",
                source_store=repo.stores[0],
            )
        if isinstance(reference, ObjectRef):
            return repo.build_object_ref(reference)
        if isinstance(reference, ConcreteDefinition):
            return repo._load_structural(reference, require_store=False, cache="none")
        if isinstance(reference, Definition):
            return repo._load_structural(
                reference.concretize(repo=repo), require_store=False, cache="none",
            )
        raise TypeError("Qualification metric reference is unsupported.")

    @staticmethod
    def _select(value, path):
        """Select one qualification tuple or mapping path."""

        components = path if isinstance(path, tuple) else (path,)
        for component in components:
            value = value[component]
        return value

    @staticmethod
    def _array(value):
        """Return one native value as a host NumPy array."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Copy one closed flat mapping into an immutable request/evidence field."""

    return MappingProxyType(dict(value))


def _case_seed(manifest: FixtureManifest, framework: str) -> dict[str, int]:
    """Return exactly the fixed seeds relevant to a matrix request."""

    initialization = manifest.baseline["initialization"]
    seed = initialization[
        {"tf": "tensorflow_seed", "torch": "torch_seed", "jax": "jax_seed"}[framework]
    ]
    if type(seed) is not int:
        raise FixtureManifestError("KTD11 initialization seed is malformed.")
    values = {"framework": seed}
    if "w3" in manifest.baseline:
        w3 = manifest.baseline["w3"]
        values.update(train=w3["train_seed"], test=w3["test_seed"])
    return values


def initialize_cpu_framework(framework: str, seed: int) -> None:
    """Establish CPU-only deterministic native initialization before model creation.

    Raises:
        QualificationUnrun: If the requested framework was already imported or
            cannot honor CPU-only deterministic controls. This protects real cases
            from inheriting an ambient GPU/device initialization.
    """

    module_name = {"tf": "tensorflow", "torch": "torch", "jax": "jax"}[framework]
    if module_name in sys.modules:
        raise QualificationUnrun(f"{module_name} was initialized before CPU qualification controls.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if framework == "tf":
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            raise QualificationUnrun("TensorFlow exposed a GPU despite CPU qualification controls.")
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    elif framework == "torch":
        import torch

        if torch.cuda.is_available():
            raise QualificationUnrun("PyTorch exposed a GPU despite CPU qualification controls.")
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    elif framework == "jax":
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        if any(device.platform != "cpu" for device in jax.devices()):
            raise QualificationUnrun("JAX exposed a non-CPU device despite CPU qualification controls.")
    else:  # pragma: no cover - QualificationCase validates the framework first.
        raise FixtureManifestError("Unsupported qualification framework.")


def selected_gpu_device(value: str | None = None) -> str:
    """Return one caller-selected physical GPU visibility selector.

    GPU qualification is deliberately bound to one numeric CUDA visibility entry.
    The isolated worker normalizes that selected physical device to native
    ``gpu:0`` evidence after it establishes visibility, so records never claim an
    unobserved physical-device spelling.

    Args:
        value: Optional caller-selected CUDA device index. When omitted, reads
            ``DRYML_ML_QUALIFICATION_GPU_DEVICE``.

    Returns:
        The nonnegative decimal device index selected by the caller.

    Raises:
        QualificationUnrun: If no single valid GPU selector was supplied before
            framework initialization.
    """

    selected = os.environ.get("DRYML_ML_QUALIFICATION_GPU_DEVICE") if value is None else value
    if type(selected) is not str or not selected.isdecimal():
        raise QualificationUnrun(
            "GPU qualification requires one caller-selected DRYML_ML_QUALIFICATION_GPU_DEVICE."
        )
    return selected


def initialize_gpu_framework(framework: str, seed: int, *, visible_device: str | None = None) -> str:
    """Establish one-GPU deterministic native initialization before model creation.

    Args:
        framework: ``"tf"`` or ``"torch"`` native backend selected by the case.
        seed: Fixed KTD11 framework seed.
        visible_device: Optional caller-selected physical CUDA device index.

    Returns:
        The normalized native GPU evidence name, always ``"gpu:0"`` after the
        selected device is made solely visible to the isolated worker.

    Raises:
        QualificationUnrun: If controls were applied too late or the selected
            worker has no usable single GPU before workload launch.
        FixtureManifestError: If ``framework`` is unsupported.

    Side Effects:
        Sets CUDA visibility before importing the selected optional framework,
        then enables that framework's deterministic controls.
    """

    if framework not in {"tf", "torch"}:
        raise FixtureManifestError("Unsupported accelerated qualification framework.")
    module_name = "tensorflow" if framework == "tf" else "torch"
    if module_name in sys.modules:
        raise QualificationUnrun(f"{module_name} was initialized before GPU qualification controls.")
    selected = selected_gpu_device(visible_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    if framework == "tf":
        import tensorflow as tf

        physical = tuple(tf.config.list_physical_devices("GPU"))
        if len(physical) != 1:
            raise QualificationUnrun("TensorFlow selected worker has no single usable GPU before launch.")
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    elif framework == "torch":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise QualificationUnrun("PyTorch selected worker has no single usable GPU before launch.")
        torch.cuda.set_device(0)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    return "gpu:0"


def preflight_gpu_framework(framework: str, *, visible_device: str | None = None) -> None:
    """Require a selected worker GPU in a disposable pre-launch probe process.

    Args:
        framework: Native framework requested by the fixed accelerated case.
        visible_device: Optional caller-selected physical CUDA device index.

    Raises:
        QualificationUnrun: If the selected framework/GPU combination is absent
            before an isolated qualification worker or output path is launched.

    Side Effects:
        Starts a short probe process with the same visibility control that the
        worker will receive. The coordinator imports no optional framework.
    """

    if framework not in {"tf", "torch"}:
        raise FixtureManifestError("Unsupported accelerated qualification framework.")
    selected = selected_gpu_device(visible_device)
    code = (
        "import sys\n"
        "name, selected = sys.argv[1:]\n"
        "if name == 'tf':\n"
        " import tensorflow as tf\n"
        " ok = len(tf.config.list_physical_devices('GPU')) == 1\n"
        "else:\n"
        " import torch\n"
        " ok = torch.cuda.is_available() and torch.cuda.device_count() == 1\n"
        "raise SystemExit(0 if ok else 3)\n"
    )
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": selected}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, framework, selected], env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise QualificationUnrun("Selected GPU worker could not complete pre-launch framework probing.") from error
    if result.returncode:
        raise QualificationUnrun("Selected GPU worker has no single usable framework GPU before launch.")


def _normalize_native_device(value: object) -> str:
    """Normalize native TF/Keras, Torch, and JAX devices to closed evidence."""

    platform = getattr(value, "platform", None)
    if platform == "cpu":
        return "cpu"
    if platform == "gpu":
        index = getattr(value, "id", None)
        if type(index) is int and index >= 0:
            return f"gpu:{index}"
        raise FixtureManifestError("Qualification observed a JAX GPU without a valid device id.")
    raw = str(value).strip().lower()
    if raw == "cpu" or "/device:cpu:" in raw or raw.startswith("cpu:"):
        return "cpu"
    if raw.startswith("cuda:"):
        index = raw.partition(":")[2]
    elif "/device:gpu:" in raw:
        index = raw.rpartition(":")[2]
    elif raw.startswith("gpu:"):
        index = raw.partition(":")[2]
    else:
        raise FixtureManifestError(f"Qualification observed an unsupported native device {raw!r}.")
    if not index.isdecimal():
        raise FixtureManifestError(f"Qualification observed an ambiguous native device {raw!r}.")
    return f"gpu:{int(index)}"


def _observed_native_device(value: object) -> object | None:
    """Return public direct or Keras-variable-handle placement metadata."""

    device = getattr(value, "device", None)
    if device is not None:
        return device() if callable(device) else device
    handle = getattr(value, "handle", None)
    return None if handle is None else getattr(handle, "device", None)


def _native_parameter_values(model) -> tuple[object, ...]:
    """Return distinct trainable native parameters from a model graph.

    Native wrappers are discovered through the public ``Repo.apply_graph``
    boundary, so composite DRYML models retain their ordinary graph semantics
    without this evidence path depending on a composite's implementation fields.
    Bare native models remain supported without creating a DRYML runtime.
    """

    direct = _native_parameter_values_for_node(model)
    if direct is not None:
        return _distinct_parameters(direct)
    try:
        from dryml.core.repo import manage_repo

        with manage_repo() as repo:
            results = repo.apply_graph(
                model,
                _native_parameter_values_for_node,
                missing="raise",
                order="post",
            )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise FixtureManifestError(
            "Qualification model exposes no traversable native TF/Keras or Torch parameters."
        ) from error
    parameters = tuple(
        parameter
        for values in results.values()
        if values is not None
        for parameter in values
    )
    if not parameters:
        raise FixtureManifestError("Qualification model exposes no native trainable parameters.")
    return _distinct_parameters(parameters)


def _native_parameter_values_for_node(model) -> tuple[object, ...] | None:
    """Return one graph node's trainable native parameters without eager imports."""

    backend = getattr(model, "native_backend", None)
    target = getattr(model, "obj", model)
    if backend == "tf":
        # The backend-owned helper recognizes built Keras variables consistently
        # with model measurement, but remains a call-time optional import.
        from dryml.tf.measurements import parameter_sets

        return parameter_sets(target)[1]
    if backend == "torch":
        from dryml.torch.measurements import parameter_sets

        return parameter_sets(target)[1]
    if backend == "jax":
        from dryml.jax.measurements import parameter_sets

        parameters = getattr(model, "parameters", ())
        effective = getattr(model, "trainable_parameters", None)
        trainables = effective("jax") if callable(effective) else parameters
        return parameter_sets(parameters, trainables)[1]
    if backend is not None:
        raise FixtureManifestError(
            f"Qualification model uses unsupported native backend {backend!r}."
        )
    # Bare native models have no DRYML backend tag. Prefer their explicit
    # trainable collection, then retain lightweight synthetic test doubles.
    trainable_variables = getattr(target, "trainable_variables", None)
    if trainable_variables is not None:
        return tuple(trainable_variables)
    parameters = getattr(target, "parameters", None)
    if callable(parameters):
        return tuple(
            parameter for parameter in parameters()
            if bool(getattr(parameter, "requires_grad", True))
        )
    variables = getattr(target, "variables", None)
    if variables is not None:
        return tuple(
            parameter for parameter in variables
            if bool(getattr(parameter, "trainable", True))
        )
    return None


def _distinct_parameters(parameters: Sequence[object]) -> tuple[object, ...]:
    """Deduplicate shared native parameters by identity while preserving order."""

    unique = {}
    for parameter in parameters:
        unique.setdefault(id(parameter), parameter)
    return tuple(unique.values())


def native_device_evidence(model, *, training_tensors: Sequence[object] = ()) -> str:
    """Return one observed device from native parameters and training tensors.

    The harness does not import optional frameworks to inspect this evidence.
    Callers must provide actual tensors consumed at the native training boundary;
    absent, unknown, or mixed parameter/tensor placement fails rather than being
    relabeled as CPU.
    """

    parameters = _native_parameter_values(model)
    values = (*parameters, *training_tensors)
    if not parameters or not training_tensors:
        raise FixtureManifestError("Qualification device evidence requires native parameters and training tensors.")
    devices = tuple(_observed_native_device(value) for value in values)
    if any(device is None for device in devices):
        raise FixtureManifestError("Qualification native parameter/training tensor device evidence is missing.")
    normalized = {_normalize_native_device(device) for device in devices}
    if len(normalized) != 1:
        raise FixtureManifestError("Qualification native parameter/training tensors have missing or mixed devices.")
    return next(iter(normalized))


def native_device_observations(
        model, *, training_tensors: Sequence[object] = (), execution_tensors: Sequence[object] = (),
) -> Mapping[str, object]:
    """Return closed placement facts from native parameters, training, and execution.

    Args:
        model: Native or DRYML-wrapped native model whose trainable parameters
            prove persisted model placement.
        training_tensors: Actual native tensors consumed at the training call.
        execution_tensors: Native results returned by the executed training call.

    Returns:
        A closed mapping containing all normalized placements and their one shared
        observed device.

    Raises:
        FixtureManifestError: If a required observation is absent, ambiguous, or
            mixed. Availability APIs alone are intentionally insufficient.
    """

    parameters = _native_parameter_values(model)
    groups = {
        "native_parameters": tuple(parameters),
        "training_tensors": tuple(training_tensors),
        "execution_tensors": tuple(execution_tensors),
    }
    if any(not values for values in groups.values()):
        raise FixtureManifestError(
            "Qualification device evidence requires native parameters, training tensors, and execution tensors."
        )
    normalized = {}
    for name, values in groups.items():
        devices = tuple(_observed_native_device(value) for value in values)
        if any(device is None for device in devices):
            raise FixtureManifestError("Qualification native device evidence is missing.")
        normalized[name] = tuple(_normalize_native_device(device) for device in devices)
    observed = {device for values in normalized.values() for device in values}
    if len(observed) != 1:
        raise FixtureManifestError("Qualification native parameters/training/execution tensors have mixed devices.")
    return MappingProxyType({**normalized, "observed_device": next(iter(observed))})


@contextmanager
def observe_jax_training_tensors(
        trainer, *, training_tensors: list[object], execution_tensors: list[object],
):
    """Observe concrete host-boundary arrays around a qualification JAX update.

    Instrumentation wraps the JITted update callable rather than the traced model
    body, so retained evidence consists of concrete device arrays instead of JAX
    tracers. Qualification workers are isolated and execute one case at a time.
    """

    def collect(destination, value):
        if isinstance(value, Mapping):
            for item in value.values():
                collect(destination, item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(destination, item)
        else:
            destination.append(value)

    trainer_type = type(trainer)
    original_update = trainer_type._native_update

    def capture_update(*args, **kwargs):
        update = original_update(*args, **kwargs)

        def observed_update(*transition_args, **transition_kwargs):
            if not training_tensors:
                collect(training_tensors, transition_args[-2:])
            result = update(*transition_args, **transition_kwargs)
            if not execution_tensors:
                collect(execution_tensors, result[0])
            return result

        return observed_update

    trainer_type._native_update = staticmethod(capture_update)
    try:
        yield
    finally:
        trainer_type._native_update = staticmethod(original_update)


def prepare_exact_model_for_formula(model, case: "QualificationCase") -> None:
    """Prepare an exact-loaded accelerated model for its requested formula device.

    Args:
        model: Exact-loaded DRYML model selected from the final Experiment receipt.
        case: Immutable qualification case identifying the native backend and
            whether the caller requested the one-GPU gate.

    Raises:
        FixtureManifestError: If an accelerated model lacks the supported runtime
            preparation API.

    Side Effects:
        Uses the model's public preparation/device API before formula inference.
        CPU cases are unchanged. This never reaches through a private native
        ``.obj`` field or chooses a physical GPU independently of the request.
    """

    if case.accelerator != "gpu":
        return
    selected_gpu_device()  # Reject a stale or missing worker selector before inference.
    if case.framework == "torch":
        import torch

        prepare = getattr(model, "to_device", None)
        if not callable(prepare):
            raise FixtureManifestError("Exact-loaded Torch formula model lacks runtime device preparation.")
        prepare(torch.device("cuda:0"))
    elif case.framework == "tf":
        prepare = getattr(model, "prep_eval", None)
        if not callable(prepare):
            raise FixtureManifestError("Exact-loaded TensorFlow formula model lacks runtime preparation.")
        prepare()
    else:  # pragma: no cover - QualificationCase validates framework values.
        raise FixtureManifestError("Accelerated formula model selects an unsupported backend.")


def _ref_to_json(reference: StateRef) -> dict[str, object]:
    """Encode one exact StateRef using the shared closed JSON grammar."""

    return _reference_to_json(reference.to_data())


def _ref_from_json(value: object) -> StateRef:
    """Decode one exact StateRef without opening its Store."""

    try:
        return StateRef.from_data(_reference_from_json(value))
    except (TypeError, ValueError) as error:
        raise FixtureManifestError("Qualification record lacks an exact StateRef.") from error


@dataclass(frozen=True, slots=True)
class QualificationCase:
    """Immutable manifest-derived request for one fixed qualification cell.

    Every request carries the selected manifest identity, fixed baseline digest,
    complete environment evidence, exact fixture identity, and seed values. It
    therefore cannot silently select a different Store/configuration later.
    """

    workload: str
    framework: str
    execution: str
    fixture_store: str
    manifest_digest: str
    config_digest: str
    environment: Mapping[str, str]
    seed: Mapping[str, int]
    w3_test_ref: StateRef | None = None
    case_kind: str = "matrix"
    tensorflow_mode: bool = False
    accelerator: str = "cpu"

    def __post_init__(self) -> None:
        """Reject incomplete, floating, or drifted request data at construction."""

        if self.workload not in WORKLOADS or self.framework not in FRAMEWORKS or self.execution not in EXECUTION_MODES:
            raise FixtureManifestError("Qualification case selects an unsupported workload, framework, or execution mode.")
        if self.case_kind not in (*CASE_KINDS, "accelerated") or type(self.tensorflow_mode) is not bool or self.accelerator not in {"cpu", "gpu"}:
            raise FixtureManifestError("Qualification case has an invalid matrix or delivery identity.")
        if self.case_kind == "matrix" and (self.tensorflow_mode or self.accelerator != "cpu"):
            raise FixtureManifestError("Primary CPU matrix cases must retain NumPy delivery.")
        if self.case_kind == "tfds-tensorflow-to-torch" and (
                (self.workload, self.framework, self.execution, self.tensorflow_mode)
                != ("W1", "torch", "local", True)):
            raise FixtureManifestError("The TFDS-to-Torch supplemental case has a fixed identity.")
        if self.case_kind == "accelerated" and (
                self.accelerator != "gpu" or self.tensorflow_mode
                or self.execution != "subprocess"
                or (self.workload, self.framework) not in {("W1", "tf"), ("W3", "torch")}):
            raise FixtureManifestError("Accelerated qualification has only the fixed TensorFlow W1 and Torch W3 GPU cases.")
        if self.case_kind != "accelerated" and self.accelerator != "cpu":
            raise FixtureManifestError("Only accelerated qualification cases may request a GPU.")
        if type(self.fixture_store) is not str or not self.fixture_store:
            raise FixtureManifestError("Qualification case lacks fixture Store authority.")
        if any(type(value) is not str or len(value) != 64 for value in (self.manifest_digest, self.config_digest)):
            raise FixtureManifestError("Qualification case identity digest is malformed.")
        if not isinstance(self.environment, Mapping) or set(self.environment) != set(REQUIRED_ENVIRONMENT_KEYS) or not all(type(value) is str and value for value in self.environment.values()):
            raise FixtureManifestError("Qualification case lacks complete environment evidence.")
        if not isinstance(self.seed, Mapping) or set(self.seed) != {"framework", "train", "test"} or any(type(value) is not int for value in self.seed.values()):
            raise FixtureManifestError("Qualification case seed evidence is malformed.")
        if self.workload == "W3":
            if type(self.w3_test_ref) is not StateRef:
                raise FixtureManifestError("W3 case lacks an exact retained test StateRef.")
        elif self.w3_test_ref is not None:
            raise FixtureManifestError("Only W3 cases may carry a W3 test StateRef.")
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
        object.__setattr__(self, "seed", _freeze_mapping(self.seed))

    @property
    def case_id(self) -> str:
        """Return stable case identity bound to all request authority fields."""

        return hashlib.sha256(json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()

    def to_data(self) -> dict[str, object]:
        """Encode the closed fresh-process-safe case request."""

        return {
            "workload": self.workload, "framework": self.framework, "execution": self.execution,
            "fixture_store": self.fixture_store, "manifest_digest": self.manifest_digest,
            "config_digest": self.config_digest, "environment": dict(self.environment),
            "seed": dict(self.seed), "w3_test_ref": None if self.w3_test_ref is None else _ref_to_json(self.w3_test_ref),
            "case_kind": self.case_kind, "tensorflow_mode": self.tensorflow_mode, "accelerator": self.accelerator,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, object]) -> "QualificationCase":
        """Decode the closed case grammar without Store or framework access."""

        required = {"workload", "framework", "execution", "fixture_store", "manifest_digest", "config_digest", "environment", "seed", "w3_test_ref", "case_kind", "tensorflow_mode", "accelerator"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise FixtureManifestError("ML workflow qualification case record is malformed.")
        reference = None if value["w3_test_ref"] is None else _ref_from_json(value["w3_test_ref"])
        try:
            return cls(
                value["workload"], value["framework"], value["execution"], value["fixture_store"],
                value["manifest_digest"], value["config_digest"], dict(value["environment"]),
                dict(value["seed"]), reference, value["case_kind"], value["tensorflow_mode"], value["accelerator"],
            )
        except (TypeError, ValueError) as error:
            raise FixtureManifestError("ML workflow qualification case record is malformed.") from error


def case_from_manifest(manifest: FixtureManifest, *, workload: str, framework: str, execution: str) -> QualificationCase:
    """Build one exact matrix request from caller-selected manifest authority."""

    case = QualificationCase(
        workload, framework, execution, str(manifest.fixture_store), manifest.digest,
        config_digest(manifest.baseline), dict(manifest.environment), _case_seed(manifest, framework),
        manifest.references.numpy if workload == "W3" else None,
    )
    if case.workload == "W3" and case.w3_test_ref != manifest.references.numpy:
        raise FixtureManifestError("W3 case changed the retained NumPy fixture StateRef.")
    return case


def supplemental_tfds_torch_case(manifest: FixtureManifest) -> QualificationCase:
    """Return the one TensorFlow-delivery-to-Torch case outside the CPU matrix."""

    return QualificationCase(
        "W1", "torch", "local", str(manifest.fixture_store), manifest.digest,
        config_digest(manifest.baseline), dict(manifest.environment), _case_seed(manifest, "torch"),
        None, "tfds-tensorflow-to-torch", True,
    )


def accelerated_cases(manifest: FixtureManifest) -> tuple[QualificationCase, QualificationCase]:
    """Return the two separately counted fixed one-GPU qualification requests.

    The cases deliberately use the existing isolated subprocess execution route,
    but their accelerator-bearing identities and gate paths cannot collide with
    the CPU matrix, local gate, supplemental delivery case, or recovery gate.
    """

    cases = tuple(
        QualificationCase(
            workload, framework, "subprocess", str(manifest.fixture_store), manifest.digest,
            config_digest(manifest.baseline), dict(manifest.environment), _case_seed(manifest, framework),
            manifest.references.numpy if workload == "W3" else None, "accelerated", False, "gpu",
        )
        for workload, framework in (("W1", "tf"), ("W3", "torch"))
    )
    if len({case.case_id for case in cases}) != 2:
        raise AssertionError("Accelerated qualification case IDs must be unique.")
    return cases


def cpu_matrix(manifest: FixtureManifest) -> tuple[QualificationCase, ...]:
    """Return the fixed 36 requests without opening fixture payloads or workers."""

    cases = tuple(
        case_from_manifest(manifest, workload=workload, framework=framework, execution=execution)
        for workload in WORKLOADS for framework in FRAMEWORKS for execution in EXECUTION_MODES
    )
    if len(cases) != 36:
        raise AssertionError("ML workflow CPU matrix must contain exactly 36 cases.")
    identities = {(case.workload, case.framework, case.execution) for case in cases}
    if len(identities) != len(cases) or identities != {
            (workload, framework, execution)
            for workload in WORKLOADS for framework in FRAMEWORKS
            for execution in EXECUTION_MODES
    }:
        raise AssertionError("ML workflow CPU matrix must enumerate every cell exactly once.")
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("ML workflow CPU matrix case IDs must be unique.")
    if any(case.tensorflow_mode or case.case_kind != "matrix" for case in cases):
        raise AssertionError("ML workflow CPU matrix must contain only NumPy-delivery matrix cases.")
    return cases


@dataclass(frozen=True, slots=True)
class QualificationCasePaths:
    """Deterministic isolated Store, work, and evidence paths for one case."""

    output_store: Path
    work_dir: Path
    evidence_dir: Path


def qualification_case_paths(
        case: QualificationCase, *, output_store_root, work_root, evidence_root,
        tfds_root, gate_id: str = "cpu-matrix", create: bool = False) -> QualificationCasePaths:
    """Derive non-replacing per-case paths below caller-provided qualification roots.

    All output roots, the read-only fixture Store, and prepared TFDS authority
    must be existing non-symlink directories with no equality or containment
    relationship. With ``create=True``, each previously absent case child is
    created once; an existing child fails closed.
    """

    roots = tuple(_safe_qualification_root(value, name) for name, value in (
        ("output Store", output_store_root), ("work", work_root),
        ("evidence", evidence_root), ("fixture Store", case.fixture_store),
        ("TFDS", tfds_root),
    ))
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if _paths_overlap(left, right):
                raise QualificationUnrun(
                    "Qualification output, fixture Store, and TFDS roots must be "
                    "disjoint without ancestor/descendant overlap."
                )
    if (
            type(gate_id) is not str or not gate_id
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in gate_id)
    ):
        raise FixtureManifestError("Qualification gate ID must be a nonempty lower-case stable name.")
    child = f"{case.case_id}--{gate_id}"
    paths = QualificationCasePaths(*(root / child for root in roots[:3]))
    if any(path.exists() for path in (paths.output_store, paths.work_dir, paths.evidence_dir)):
        raise FixtureManifestError("Refusing to replace an existing qualification case path.")
    if create:
        for path in (paths.output_store, paths.work_dir, paths.evidence_dir):
            path.mkdir()
    return paths


def _safe_qualification_root(value, name: str) -> Path:
    """Resolve one existing authority root while rejecting symlink ambiguity."""

    raw = Path(value).expanduser()
    raw = raw if raw.is_absolute() else Path.cwd() / raw
    current = Path(raw.anchor)
    try:
        for component in raw.parts[1:]:
            current /= component
            if current.is_symlink():
                raise QualificationUnrun(
                    f"Qualification {name} root contains an ambiguous symlink."
                )
        root = raw.resolve(strict=True)
    except OSError as error:
        raise QualificationUnrun(
            f"Qualification {name} root must be an existing accessible directory."
        ) from error
    if not root.is_dir():
        raise QualificationUnrun(
            f"Qualification {name} root must be an existing accessible directory."
        )
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether resolved roots are equal or ancestor-related."""

    return left == right or left in right.parents or right in left.parents


def mnist_pipeline(source, *, autoencode: bool = False, label_dtype: str | None = None):
    """Build the public U4-U6 MNIST Method graph from a supplied TFDS Dataset.

    Image projection, normalization, flatten/layout adaptation, cast, and target
    projection are all graph nodes. ``label_dtype`` lets the caller author a
    backend-supported sparse-label dtype without relying on implicit narrowing.
    No caller-written backend conversion is used.
    """

    image = Pipe(Select(0), ImageNormalize(), Flatten(), Cast("float32"))
    target = image if autoencode else Select(1)
    if label_dtype is not None and not autoencode:
        target = Pipe(target, Cast(label_dtype))
    return Map(
        source,
        Project(image, target),
        preserves_examples=True,
    )


def w1_label_methods():
    """Return graph-visible W1 prediction/target decoding Methods."""

    return {"prediction": ArgMax(axis=-1), "target": Select()}


def polynomial_samples(*, seed: int, count: int) -> ArrayDataset:
    """Return fixed W3 noisy observations for fixture preparation or training."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(count, 1)).astype("float32")
    truth = 0.3 + 0.7 * x - 0.5 * x ** 2
    return ArrayDataset({"x": x, "y": (truth + rng.normal(0.0, 0.05, size=x.shape)).astype("float32")})


def build_w3_fixtures(store, baseline: Mapping[str, object]):
    """Build distinct completed NumPy/Parquet W3 caches in caller-selected Store.

    This explicit preparation builder performs no network work. It is intended for
    :func:`prepare_manifest`, which validates completion and equivalence before it
    publishes final manifest authority.
    """

    from dryml.artifacts import CachedDataset
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed import ManagedConfig
    from .ml_workflow_fixtures import FixtureReferences

    w3 = baseline["w3"]
    repo = Repo(DirStore(store))
    source = repo.save_object(polynomial_samples(seed=w3["test_seed"], count=w3["test_count"]))
    config = ManagedConfig(state_repo=repo)
    return FixtureReferences(
        CachedDataset(source).compute(codec="numpy", managed=config),
        CachedDataset(source).compute(codec="parquet", managed=config),
    )


def accuracy_formula(predictions, targets) -> float:
    """Compute W1's independent decoded categorical-accuracy formula."""

    predicted, expected = np.asarray(predictions), np.asarray(targets)
    if predicted.shape != expected.shape or predicted.size == 0:
        raise ValueError("W1 accuracy requires equally shaped nonempty decoded labels.")
    return float(np.mean(predicted == expected))


def mse_formula(predictions, observations) -> float:
    """Compute W2/W3's independent mean squared residual formula."""

    predicted, expected = np.asarray(predictions, dtype=np.float64), np.asarray(observations, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.size == 0:
        raise ValueError("MSE requires equally shaped nonempty predictions and observations.")
    result = float(np.mean((predicted - expected) ** 2))
    if not math.isfinite(result):
        raise ValueError("MSE inputs must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    """Closed KTD11 result record bound to one manifest-derived request.

    The record carries final Experiment/model/test authority, completed history and
    Artifact receipts, independent formula provenance, environment/runtime facts,
    and bounded resource measurements. Unknown, missing, non-finite, and
    inconsistent data is rejected before a result is accepted.
    """

    case: QualificationCase
    final_experiment_ref: StateRef
    model_ref: StateRef
    test_ref: StateRef
    history_ref: StateRef
    history_rows: tuple[Mapping[str, object], ...]
    artifact_ref: StateRef
    artifact_value: float
    formula: Mapping[str, object]
    environment: Mapping[str, str]
    runtime: Mapping[str, object]
    elapsed_seconds: float
    peak_rss_bytes: int
    output_bytes: int
    worker: Mapping[str, object]
    recovery: Mapping[str, object] | None = None
    device_evidence: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate closed field types before any result can be serialized."""

        if not isinstance(self.case, QualificationCase):
            raise FixtureManifestError("Qualification evidence lacks a case identity.")
        if any(type(reference) is not StateRef for reference in (self.final_experiment_ref, self.model_ref, self.test_ref, self.history_ref, self.artifact_ref)):
            raise FixtureManifestError("Qualification evidence requires exact StateRefs.")
        if not self.history_rows or any(not isinstance(row, Mapping) for row in self.history_rows):
            raise FixtureManifestError("Qualification evidence requires nonempty history rows.")
        if not all(type(value) in (int, float) and math.isfinite(float(value)) for value in (self.artifact_value, self.elapsed_seconds)):
            raise FixtureManifestError("Qualification evidence numeric values must be finite.")
        if type(self.peak_rss_bytes) is not int or type(self.output_bytes) is not int or self.peak_rss_bytes < 0 or self.output_bytes < 0 or self.elapsed_seconds < 0:
            raise FixtureManifestError("Qualification evidence resource values must be nonnegative integers.")
        if set(self.formula) != {"name", "value", "provenance"} or self.formula["name"] != _FORMULAS[self.case.workload] or type(self.formula["provenance"]) is not str or not self.formula["provenance"]:
            raise FixtureManifestError("Qualification formula evidence is malformed.")
        if type(self.formula["value"]) not in (int, float) or not math.isfinite(float(self.formula["value"])):
            raise FixtureManifestError("Qualification formula value must be finite.")
        if dict(self.environment) != dict(self.case.environment):
            raise FixtureManifestError("Qualification evidence environment does not match its request.")
        if set(self.runtime) != {"backend", "device", "worker_id", "process_id"} or type(self.runtime["backend"]) is not str or type(self.runtime["device"]) is not str or type(self.runtime["worker_id"]) is not str or type(self.runtime["process_id"]) is not int:
            raise FixtureManifestError("Qualification runtime evidence is incomplete.")
        if self.runtime["backend"] != self.case.framework or not self.runtime["device"] or not self.runtime["worker_id"] or self.runtime["process_id"] < 0:
            raise FixtureManifestError("Qualification runtime evidence disagrees with its request.")
        self._validate_device_evidence()
        self._validate_worker_evidence()
        self._validate_recovery_evidence()
        expected_metric = _METRIC_NAMES[self.case.workload]
        for row in self.history_rows:
            if set(row) != {"state_ref", "evaluation_status", "eval_artifacts"} or row["state_ref"] != self.final_experiment_ref or row["evaluation_status"] != "completed":
                raise FixtureManifestError("Qualification history evidence is incomplete or misassociated.")
            artifacts = row["eval_artifacts"]
            if not isinstance(artifacts, Mapping) or artifacts != {expected_metric: self.artifact_ref}:
                raise FixtureManifestError("Qualification history Artifact evidence is incomplete or misassociated.")
        object.__setattr__(
            self,
            "history_rows",
            tuple(MappingProxyType({
                "state_ref": row["state_ref"], "evaluation_status": row["evaluation_status"],
                "eval_artifacts": MappingProxyType(dict(row["eval_artifacts"])),
            }) for row in self.history_rows),
        )
        object.__setattr__(self, "formula", _freeze_mapping(self.formula))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
        object.__setattr__(self, "runtime", _freeze_mapping(self.runtime))
        object.__setattr__(self, "worker", _freeze_mapping(self.worker))
        if self.recovery is not None:
            object.__setattr__(self, "recovery", _freeze_mapping(self.recovery))

    def _validate_device_evidence(self) -> None:
        """Require claimed accelerated placement to equal actual native observations."""

        if self.case.accelerator == "cpu":
            if self.device_evidence is not None:
                raise FixtureManifestError("CPU qualification must not claim accelerated device evidence.")
            if self.runtime["device"] != "cpu":
                raise FixtureManifestError("CPU qualification runtime evidence must remain on CPU.")
            return
        required = {"allocation", "native_parameters", "training_tensors", "execution_tensors", "observed_device"}
        evidence = self.device_evidence
        if not isinstance(evidence, Mapping) or set(evidence) != required:
            raise FixtureManifestError("Accelerated qualification device evidence is incomplete.")
        allocation = evidence["allocation"]
        if not isinstance(allocation, Mapping) or set(allocation) != {"visible_device", "claimed_device"}:
            raise FixtureManifestError("Accelerated qualification allocation evidence is incomplete.")
        if type(allocation["visible_device"]) is not str or not allocation["visible_device"].isdecimal() or allocation["claimed_device"] != "gpu:0":
            raise FixtureManifestError("Accelerated qualification allocation evidence is invalid.")
        groups = ("native_parameters", "training_tensors", "execution_tensors")
        if any(not isinstance(evidence[name], (tuple, list)) or not evidence[name] or any(item != "gpu:0" for item in evidence[name]) for name in groups):
            raise FixtureManifestError("Accelerated qualification native placement evidence is missing, mixed, or non-GPU.")
        if evidence["observed_device"] != "gpu:0" or self.runtime["device"] != evidence["observed_device"]:
            raise FixtureManifestError("Accelerated qualification claimed GPU disagrees with observed native placement.")
        object.__setattr__(self, "device_evidence", MappingProxyType({
            "allocation": MappingProxyType(dict(allocation)),
            **{name: tuple(evidence[name]) for name in groups},
            "observed_device": evidence["observed_device"],
        }))

    def _validate_worker_evidence(self) -> None:
        """Require detached Core Execute route, authority, and receipt facts."""

        required = {
            "execution_backend", "semantic_backend", "isolation", "worker_pid",
            "worker_identity", "runtime_allocation", "shared_authority", "submitted",
            "core_outcome", "final_refs", "coordinator_validated_refs",
        }
        if not isinstance(self.worker, Mapping) or set(self.worker) != required:
            raise FixtureManifestError("Qualification worker evidence is incomplete.")
        backend = self.worker["execution_backend"]
        expected_backend = {
            "local": "isolation-process", "managed-local": "isolation-process",
            "subprocess": "subprocess", "ray": "ray",
        }[self.case.execution]
        provisional = self.worker["coordinator_validated_refs"] is None
        if backend != expected_backend and not (provisional and backend == "worker-provisional"):
            raise FixtureManifestError("Qualification worker backend does not match its observed route.")
        expected_semantic = "local" if self.case.execution in {"local", "managed-local"} else self.case.execution
        if self.worker["semantic_backend"] != expected_semantic:
            raise FixtureManifestError("Qualification worker semantic backend is inconsistent.")
        isolation = self.worker["isolation"]
        if not isinstance(isolation, Mapping) or set(isolation) != {"kind", "parent_pid"} or not isinstance(isolation["kind"], str) or type(isolation["parent_pid"]) is not int:
            raise FixtureManifestError("Qualification worker isolation evidence is malformed.")
        if (self.case.execution in {"local", "managed-local"} and not provisional
                and isolation["kind"] != "fresh-os-process"):
            raise FixtureManifestError("Local qualification evidence lacks a fresh isolation process.")
        if type(self.worker["worker_pid"]) is not int or self.worker["worker_pid"] < 0 or not isinstance(self.worker["worker_identity"], str) or not self.worker["worker_identity"]:
            raise FixtureManifestError("Qualification worker identity evidence is malformed.")
        allocation = self.worker["runtime_allocation"]
        if not isinstance(allocation, Mapping) or set(allocation) != {"resource_mode", "allocation", "admission"} or not all(isinstance(allocation[name], str) and allocation[name] for name in allocation):
            raise FixtureManifestError("Qualification worker allocation evidence is malformed.")
        if not provisional and self.case.execution in {"local", "managed-local"}:
            expected_mode = "managed-session" if self.case.execution == "managed-local" else "unmanaged-session"
            if allocation["resource_mode"] != expected_mode:
                raise FixtureManifestError("Local qualification allocation evidence disagrees with observed session mode.")
            if self.case.execution == "local" and allocation["allocation"] != "no-session-allocation":
                raise FixtureManifestError("Unmanaged local qualification reported a session allocation.")
            if self.case.execution == "managed-local" and allocation["allocation"] == "no-session-allocation":
                raise FixtureManifestError("Managed local qualification omitted its session allocation.")
        if not provisional and self.case.execution in {"subprocess", "ray"}:
            expected = (
                {
                    "resource_mode": "worker-process-one-gpu",
                    "allocation": "worker-process-one-gpu",
                    "admission": "admitted-with-one-gpu-visibility",
                }
                if self.case.accelerator == "gpu" else
                {
                    "resource_mode": "worker-process-no-session-allocation",
                    "allocation": "worker-process-no-session-allocation",
                    "admission": "admitted-without-session-allocation",
                }
            )
            if allocation != expected:
                raise FixtureManifestError("Core worker allocation evidence disagrees with its no-session route.")
        authority = self.worker["shared_authority"]
        if not isinstance(authority, Mapping) or set(authority) != {"fixture_store", "control_store"} or not all(isinstance(value, str) and value for value in authority.values()):
            raise FixtureManifestError("Qualification worker shared authority evidence is malformed.")
        submitted = self.worker["submitted"]
        if not isinstance(submitted, Mapping) or set(submitted) != {"request_digest", "submission_id", "result_ref"} or not all(isinstance(submitted[name], str) and submitted[name] for name in submitted) or submitted["result_ref"] != self.final_experiment_ref.digest():
            raise FixtureManifestError("Qualification worker submission receipts are inconsistent.")
        outcome = self.worker["core_outcome"]
        if not isinstance(outcome, Mapping) or set(outcome) != {"kind", "publication_refs", "update_refs"} or not isinstance(outcome["kind"], str) or not outcome["kind"] or not isinstance(outcome["publication_refs"], (tuple, list)) or not isinstance(outcome["update_refs"], (tuple, list)) or not all(isinstance(reference, str) and reference for reference in outcome["publication_refs"]) or not all(isinstance(reference, str) and reference for reference in outcome["update_refs"]):
            raise FixtureManifestError("Qualification Core outcome evidence is malformed.")
        final_refs = self.worker["final_refs"]
        expected_refs = {
            "experiment": self.final_experiment_ref.digest(), "model": self.model_ref.digest(),
            "test": self.test_ref.digest(), "history": self.history_ref.digest(),
            "artifact": self.artifact_ref.digest(),
        }
        if not isinstance(final_refs, Mapping) or dict(final_refs) != expected_refs:
            raise FixtureManifestError("Qualification worker final reference evidence is inconsistent.")
        coordinator_refs = self.worker["coordinator_validated_refs"]
        if coordinator_refs is not None:
            if not isinstance(coordinator_refs, Mapping) or set(coordinator_refs) != {"output_store", "refs"}:
                raise FixtureManifestError("Qualification coordinator final-reference evidence is malformed.")
            if not isinstance(coordinator_refs["output_store"], str) or not coordinator_refs["output_store"]:
                raise FixtureManifestError("Qualification coordinator output Store evidence is malformed.")
            if not isinstance(coordinator_refs["refs"], Mapping) or dict(coordinator_refs["refs"]) != expected_refs:
                raise FixtureManifestError("Qualification coordinator final-reference evidence is inconsistent.")

    def _validate_recovery_evidence(self) -> None:
        """Validate optional fixed step-64 repair evidence for the recovery case."""

        if self.recovery is None:
            return
        required = {"checkpoint_step", "before", "restored", "events", "final_refs"}
        if (self.case.workload, self.case.framework, self.case.execution) != ("W3", "torch", "subprocess") or not isinstance(self.recovery, Mapping) or set(self.recovery) != required or self.recovery["checkpoint_step"] != 64:
            raise FixtureManifestError("Qualification recovery evidence is malformed.")
        retained_fields = {"model_digest", "model_placement", "optimizer_digest", "optimizer_placement", "optimizer_iterations", "epoch", "next_batch", "step", "examples_seen", "loss_numerator", "loss_denominator", "checkpoint_ref", "history_occurrence", "artifact_status"}
        before, restored = self.recovery["before"], self.recovery["restored"]
        if not isinstance(before, Mapping) or not isinstance(restored, Mapping) or set(before) != retained_fields or set(restored) != retained_fields or before != restored:
            raise FixtureManifestError("Qualification recovery evidence does not retain exact restored authority facts.")
        events = self.recovery["events"]
        if not isinstance(events, (tuple, list)) or tuple(events)[:2] != ("artifact_repaired", "optimizer_update:65") or len(set(events)) != len(events):
            raise FixtureManifestError("Qualification recovery did not repair the pending Artifact before update 65.")
        if self.recovery["final_refs"] != self.worker["final_refs"]:
            raise FixtureManifestError("Qualification recovery final references disagree with worker evidence.")

    def to_data(self) -> dict[str, object]:
        """Encode a machine-readable closed evidence record."""

        def row_data(row):
            return {
                "state_ref": _ref_to_json(row["state_ref"]), "evaluation_status": row["evaluation_status"],
                "eval_artifacts": {name: _ref_to_json(reference) for name, reference in row["eval_artifacts"].items()},
            }
        return {
            "case": self.case.to_data(), "final_experiment_ref": _ref_to_json(self.final_experiment_ref),
            "model_ref": _ref_to_json(self.model_ref), "test_ref": _ref_to_json(self.test_ref),
            "history_ref": _ref_to_json(self.history_ref), "history_rows": [row_data(row) for row in self.history_rows],
            "artifact_ref": _ref_to_json(self.artifact_ref), "artifact_value": self.artifact_value,
            "formula": dict(self.formula), "environment": dict(self.environment), "runtime": dict(self.runtime),
            "elapsed_seconds": self.elapsed_seconds, "peak_rss_bytes": self.peak_rss_bytes, "output_bytes": self.output_bytes,
            "worker": dict(self.worker), "recovery": None if self.recovery is None else dict(self.recovery),
            "device_evidence": None if self.device_evidence is None else {
                "allocation": dict(self.device_evidence["allocation"]),
                "native_parameters": list(self.device_evidence["native_parameters"]),
                "training_tensors": list(self.device_evidence["training_tensors"]),
                "execution_tensors": list(self.device_evidence["execution_tensors"]),
                "observed_device": self.device_evidence["observed_device"],
            },
        }

    @classmethod
    def from_data(cls, value: Mapping[str, object]) -> "QualificationEvidence":
        """Decode the closed evidence grammar without opening a Store or framework.

        Args:
            value: JSON-decoded evidence object created by :meth:`to_data`.

        Returns:
            A construction-validated immutable evidence record.

        Raises:
            FixtureManifestError: If fields are missing, extra, or contain invalid
                exact-reference/history associations.
        """

        required = {
            "case", "final_experiment_ref", "model_ref", "test_ref", "history_ref", "history_rows",
            "artifact_ref", "artifact_value", "formula", "environment", "runtime", "elapsed_seconds",
            "peak_rss_bytes", "output_bytes", "worker", "recovery", "device_evidence",
        }
        if not isinstance(value, Mapping) or set(value) != required or not isinstance(value["history_rows"], list):
            raise FixtureManifestError("Qualification evidence record has missing or extra fields.")
        rows = []
        for row in value["history_rows"]:
            if not isinstance(row, Mapping) or set(row) != {"state_ref", "evaluation_status", "eval_artifacts"} or not isinstance(row["eval_artifacts"], Mapping):
                raise FixtureManifestError("Qualification evidence history row is malformed.")
            try:
                rows.append({
                    "state_ref": _ref_from_json(row["state_ref"]), "evaluation_status": row["evaluation_status"],
                    "eval_artifacts": {name: _ref_from_json(reference) for name, reference in row["eval_artifacts"].items()},
                })
            except (TypeError, ValueError) as error:
                raise FixtureManifestError("Qualification evidence history references are malformed.") from error
        try:
            return cls(
                QualificationCase.from_data(value["case"]), _ref_from_json(value["final_experiment_ref"]),
                _ref_from_json(value["model_ref"]), _ref_from_json(value["test_ref"]),
                _ref_from_json(value["history_ref"]), tuple(rows), _ref_from_json(value["artifact_ref"]),
                value["artifact_value"], dict(value["formula"]), dict(value["environment"]),
                dict(value["runtime"]), value["elapsed_seconds"], value["peak_rss_bytes"], value["output_bytes"],
                dict(value["worker"]), None if value["recovery"] is None else dict(value["recovery"]),
                None if value["device_evidence"] is None else dict(value["device_evidence"]),
            )
        except (TypeError, ValueError) as error:
            raise FixtureManifestError("Qualification evidence record is malformed.") from error


def validate_evidence(
        manifest: FixtureManifest, evidence: QualificationEvidence, *, output_store=None,
        coordinator_results=None) -> None:
    """Validate all KTD11 association, formula, threshold, and resource gates.

    ``coordinator_results`` is the already scoped final Experiment receipt,
    ExperimentData history, and scalar Artifact result.  It keeps orchestrator
    validation from materializing the referenced model or test Dataset payload.
    Standalone local qualification leaves it ``None`` and performs its ordinary
    non-orchestrator full result validation.
    """

    case = evidence.case
    if case.fixture_store != str(manifest.fixture_store) or case.manifest_digest != manifest.digest or case.config_digest != config_digest(manifest.baseline):
        raise FixtureManifestError("Qualification evidence is not bound to manifest authority.")
    if case.workload == "W3" and case.w3_test_ref != manifest.references.numpy:
        raise FixtureManifestError("W3 evidence changed the retained fixture StateRef.")
    try:
        expected_model = evidence.final_experiment_ref.at("model")
        expected_test = evidence.final_experiment_ref.reference_value_at("test_data")
    except ValueError as error:
        raise FixtureManifestError("Final Experiment StateRef lacks exact model/test bindings.") from error
    if expected_model != evidence.model_ref or expected_test != evidence.test_ref:
        raise FixtureManifestError("Evidence model/test bindings disagree with final Experiment StateRef.")
    if case.workload == "W3" and evidence.test_ref != manifest.references.numpy:
        raise FixtureManifestError("W3 Experiment did not retain the manifest NumPy cache StateRef.")
    coordinator_refs = evidence.worker["coordinator_validated_refs"]
    if coordinator_refs is None:
        raise FixtureManifestError("Worker provisional evidence has no coordinator final-reference authority.")
    if output_store is None or coordinator_refs["output_store"] != os.fspath(Path(output_store).resolve()):
        raise FixtureManifestError("Coordinator final-reference authority names another output Store.")
    metric_name = _METRIC_NAMES[case.workload]
    for row in evidence.history_rows:
        if set(row) != {"state_ref", "evaluation_status", "eval_artifacts"} or row["state_ref"] != evidence.final_experiment_ref or row["evaluation_status"] != "completed":
            raise FixtureManifestError("Qualification history row is incomplete or bound to another checkpoint.")
        if row["eval_artifacts"] != {metric_name: evidence.artifact_ref}:
            raise FixtureManifestError("Qualification history Artifact receipt is incomplete or inconsistent.")
    if not math.isclose(float(evidence.artifact_value), float(evidence.formula["value"]), rel_tol=1e-6, abs_tol=1e-7):
        raise FixtureManifestError("Artifact value does not match the independent formula.")
    budget_seconds = manifest.baseline["cpu_budget_seconds"]
    rss_budget = manifest.baseline["cpu_peak_rss_bytes"]
    output_budget = manifest.baseline["case_store_budget_bytes"]
    if evidence.elapsed_seconds > budget_seconds or evidence.peak_rss_bytes > rss_budget or evidence.output_bytes > output_budget:
        raise FixtureManifestError("Qualification evidence exceeds a fixed KTD11 resource budget.")
    failed_gate = evidence.artifact_value < THRESHOLDS["W1"] if case.workload == "W1" else evidence.artifact_value > THRESHOLDS[case.workload]
    if failed_gate:
        raise FixtureManifestError(f"{case.workload} did not meet its fixed acceptance threshold.")
    # Validate retained history/result records rather than trusting runner-provided
    # row copies. The coordinator supplies only its explicitly scoped lightweight
    # loads; standalone local qualification retains its pre-existing full check.
    if coordinator_results is None:
        from dryml.artifacts import Value
        from dryml.core import Repo
        from dryml.core.store.dir import DirStore
        from dryml.data import Dataset
        from dryml.models import Experiment, ExperimentData, Model

        stores = (DirStore(output_store), DirStore(manifest.fixture_store)) if output_store is not None else DirStore(manifest.fixture_store)
        repo = Repo(stores)
        try:
            final_experiment = repo.load_state_ref(evidence.final_experiment_ref, reuse_live="never", cache="none")
            model = repo.load_state_ref(evidence.model_ref, reuse_live="never", cache="none")
            test_data = repo.load_state_ref(evidence.test_ref, reuse_live="never", cache="none")
            history = repo.load_state_ref(evidence.history_ref, reuse_live="never", cache="none")
            artifact = repo.load_state_ref(evidence.artifact_ref, reuse_live="never", cache="none")
        except Exception as error:
            raise FixtureManifestError("Qualification StateRef authority is absent or corrupt in the selected Store.") from error
        if not isinstance(final_experiment, Experiment) or not isinstance(model, Model) or not isinstance(test_data, Dataset):
            raise FixtureManifestError("Qualification final Experiment, projected model, or test Dataset type is invalid.")
        if not isinstance(history, ExperimentData) or not isinstance(artifact, Value) or not artifact.ready:
            raise FixtureManifestError("Qualification history or Artifact completion evidence is invalid.")
        if (
                final_experiment.last_state_ref != evidence.final_experiment_ref
                or model.last_state_ref != evidence.model_ref
                or test_data.last_state_ref != evidence.test_ref
                or history.last_state_ref != evidence.history_ref
                or artifact.last_state_ref != evidence.artifact_ref):
            raise FixtureManifestError("Qualification receipt does not equal exact selected Store authority.")
        if (
                final_experiment.last_state_ref.at("model") != evidence.model_ref
                or final_experiment.last_state_ref.reference_value_at("test_data") != evidence.test_ref):
            raise FixtureManifestError("Loaded final Experiment bindings disagree with evidence authority.")
        observed_parameters = _native_parameter_values(model)
        observed_devices = {
            _normalize_native_device(device)
            for parameter in observed_parameters
            if (device := _observed_native_device(parameter)) is not None
        }
        if len(observed_devices) != 1 or evidence.runtime["device"] != next(iter(observed_devices)):
            raise FixtureManifestError("Qualification evidence device disagrees with saved native parameter placement.")
    else:
        if not isinstance(coordinator_results, tuple) or len(coordinator_results) != 3:
            raise FixtureManifestError("Coordinator result validation must supply exactly three selected results.")
        _, history, artifact = coordinator_results
    stored_rows = history.data.to_dict("records")
    expected_rows = [
        {
            "state_ref": row["state_ref"], "evaluation_status": row["evaluation_status"],
            "eval_artifacts": dict(row["eval_artifacts"]),
        }
        for row in evidence.history_rows
    ]
    observed_rows = [
        {
            "state_ref": record.get("state_ref"), "evaluation_status": record.get("evaluation_status"),
            "eval_artifacts": record.get("eval_artifacts"),
        }
        for record in stored_rows
        if record.get("state_ref") == evidence.final_experiment_ref
    ]
    if observed_rows != expected_rows:
        raise FixtureManifestError("Persisted history rows do not exactly equal the evidenced final receipts.")
    try:
        stored_value = float(artifact.value())
    except (TypeError, ValueError) as error:
        raise FixtureManifestError("Persisted Artifact has no scalar qualification metric.") from error
    if not math.isfinite(stored_value) or not math.isclose(stored_value, evidence.artifact_value, rel_tol=1e-6, abs_tol=1e-7):
        raise FixtureManifestError("Persisted Artifact value disagrees with evidence.")


def require_real_qualification(opted_in: bool, *, prerequisites: Sequence[str] = ()) -> None:
    """Report disabled or unavailable real work as truthful unrun status."""

    if not opted_in:
        raise QualificationUnrun("Real ML workflow qualification requires explicit opt-in.")
    if prerequisites:
        raise QualificationUnrun("ML workflow qualification prerequisites are unavailable: " + ", ".join(prerequisites))


def run_local_case(manifest: FixtureManifest, case: QualificationCase, *, opted_in: bool, runner: Callable[[QualificationCase], QualificationEvidence], prerequisites: Sequence[str] = (), output_store=None) -> QualificationEvidence:
    """Preflight and run one selected local/managed-local real workload.

    The runner receives only an immutable manifest-derived request. It is never
    called without explicit opt-in and successful persistent fixture preflight.
    """

    require_real_qualification(opted_in, prerequisites=prerequisites)
    if case.execution not in {"local", "managed-local"}:
        raise FixtureManifestError("run_local_case accepts only local execution cases.")
    if output_store is None:
        raise QualificationUnrun("Local qualification requires a caller-selected case output Store.")
    if _absolute_output_store(output_store) == manifest.fixture_store:
        raise FixtureManifestError("Qualification output Store must differ from the fixture Store.")
    preflight_manifest(manifest)
    if not callable(runner):
        raise TypeError("runner must be a real qualification callback.")
    evidence = runner(case)
    if not isinstance(evidence, QualificationEvidence):
        raise FixtureManifestError("Local qualification runner did not return closed evidence.")
    validate_evidence(
        manifest, evidence,
        output_store=_absolute_output_store(output_store) / f"{case.case_id}--local-qualification",
    )
    return evidence


def _absolute_output_store(value):
    """Normalize one caller-selected local output Store path without creating it."""

    from pathlib import Path

    return Path(value).expanduser().resolve(strict=False)


def _model_and_training(framework: str, workload: str, *, seed: int):
    """Construct the fixed native model/trainer pair only in a selected real run."""

    from dryml import F
    from dryml.models import AutoEncoder
    if framework == "torch":
        import torch
        from dryml.models.torch import Optimizer, Sequential, Training
        if workload == "W1":
            model = Sequential(layer_defs=(F("Linear", 784, 128), F("ReLU"), F("Linear", 128, 10)))
            trainer = Training(optimizer=Optimizer(torch.optim.Adam, target=model, lr=1e-3), loss_cls=torch.nn.CrossEntropyLoss, epochs=1, verbose=0)
        elif workload == "W2":
            model = AutoEncoder(Sequential(layer_defs=(F("Linear", 784, 32), F("ReLU"))), Sequential(layer_defs=(F("Linear", 32, 784), F("Sigmoid"))))
            trainer = Training(optimizer=Optimizer(torch.optim.Adam, target=model, lr=1e-3), loss_cls=torch.nn.MSELoss, epochs=1, verbose=0)
        else:
            model = Sequential(layer_defs=(F("Linear", 1, 32), F("Tanh"), F("Linear", 32, 1)))
            trainer = Training(optimizer=Optimizer(torch.optim.Adam, target=model, lr=1e-3), loss_cls=torch.nn.MSELoss, epochs=10, verbose=0)
        return model, trainer
    if framework == "tf":
        import tensorflow as tf
        from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential
        if workload == "W1":
            model = Sequential(layer_defs=(F("Dense", units=128, activation="relu"), F("Dense", units=10)))
            trainer = BasicTraining(optimizer=Optimizer(tf.keras.optimizers.Adam, learning_rate=1e-3), loss=Loss(tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True), epochs=1, verbose=0)
        elif workload == "W2":
            model = AutoEncoder(Sequential(layer_defs=(F("Dense", units=32, activation="relu"),)), Sequential(layer_defs=(F("Dense", units=784, activation="sigmoid"),)))
            trainer = BasicTraining(optimizer=Optimizer(tf.keras.optimizers.Adam, learning_rate=1e-3), loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, verbose=0)
        else:
            model = Sequential(layer_defs=(F("Dense", units=32, activation="tanh"), F("Dense", units=1)))
            trainer = BasicTraining(optimizer=Optimizer(tf.keras.optimizers.Adam, learning_rate=1e-3), loss=Loss(tf.keras.losses.MeanSquaredError), epochs=10, verbose=0)
        return model, trainer
    if framework == "jax":
        from dryml.core import TensorSpec
        from dryml.models.jax import Model, Optimizer, Training

        if workload == "W1":
            widths, hidden, output, epochs = (784, 128, 10), "relu", None, 1
            loss = _jax_sparse_categorical_loss
        elif workload == "W2":
            widths, hidden, output, epochs = (784, 32, 784), "relu", "sigmoid", 1
            loss = _jax_mse
        else:
            widths, hidden, output, epochs = (1, 32, 1), "tanh", None, 10
            loss = _jax_mse
        model = Model(
            F(_jax_mlp_init_factory),
            F(_jax_mlp_apply_factory, hidden_activation=hidden, output_activation=output),
            widths,
            output_spec=TensorSpec("float32", shape=(widths[-1],), backend="jax"),
            seed=seed,
        )
        trainer = Training(
            optimizer=Optimizer(F(_jax_adam_factory, learning_rate=1e-3)),
            loss=loss,
            epochs=epochs,
            verbose=0,
        )
        return model, trainer
    raise FixtureManifestError("Unsupported qualification framework.")


def _jax_mlp_init(key, widths):
    """Initialize a deterministic dense JAX network for selected qualification."""

    import jax

    keys = jax.random.split(key, len(widths) - 1)
    layers = []
    for layer_key, input_width, output_width in zip(keys, widths[:-1], widths[1:]):
        limit = math.sqrt(6.0 / (input_width + output_width))
        layers.append({
            "weight": jax.random.uniform(
                layer_key, (input_width, output_width), minval=-limit,
                maxval=limit, dtype=jax.numpy.float32,
            ),
            "bias": jax.numpy.zeros((output_width,), dtype=jax.numpy.float32),
        })
    return layers, {}


def _jax_mlp_apply(
        parameters, mutable_state, rng, value, training, *,
        hidden_activation, output_activation):
    """Apply the qualification dense network as one pure JAX candidate."""

    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    result = value
    for index, layer in enumerate(parameters):
        result = result @ layer["weight"] + layer["bias"]
        activation = output_activation if index == len(parameters) - 1 else hidden_activation
        if activation == "relu":
            result = jax.nn.relu(result)
        elif activation == "tanh":
            result = jax.numpy.tanh(result)
        elif activation == "sigmoid":
            result = jax.nn.sigmoid(result)
    return result, mutable_state, next_rng


def _jax_mlp_init_factory():
    """Build the definition-compatible qualification initializer."""

    return _jax_mlp_init


def _jax_mlp_apply_factory(*, hidden_activation, output_activation):
    """Build the definition-compatible qualification apply callable."""

    from functools import partial

    return partial(
        _jax_mlp_apply,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
    )


def _jax_adam_factory(*, learning_rate):
    """Build the qualification Optax optimizer lazily in a selected worker."""

    import optax

    return optax.adam(learning_rate)


def _jax_mse(predictions, targets):
    """Return scalar mean squared error for JAX qualification."""

    import jax

    return jax.numpy.mean((predictions - targets) ** 2)


def _jax_sparse_categorical_loss(predictions, targets):
    """Return scalar sparse softmax loss for JAX qualification."""

    import optax

    return optax.softmax_cross_entropy_with_integer_labels(predictions, targets).mean()


def build_workload(
        repo, case: QualificationCase, *, managed: ManagedConfig,
        mnist_source: Callable[[str, bool], object] | None = None,
):
    """Build real Experiment W1/W2/W3 workflows from a manifest-derived request.

    Args:
        repo: Isolated case output Repo receiving model, Dataset, and result state.
        case: Closed manifest-derived workload and backend selection.
        managed: Managed authority selecting the output Repo and request control
            Store for W1/W2 cache publication.
        mnist_source: Prepared no-download TFDS source factory required by W1/W2.

    Returns:
        A configured Experiment whose metric recipe binds to exact checkpoint
        model and test Dataset references.

    Raises:
        QualificationUnrun: If W1/W2 has no prepared source or native framework
            initialization cannot honor the selected case.
        FixtureManifestError: If the case selects an unsupported framework.

    Side Effects:
        Initializes the selected native framework and publishes the fixed W1/W2
        test slice to the output Repo using the request's control Store.

    W1/W2 require a caller-provided already-prepared TFDS source factory. W3 uses
    the exact manifest StateRef embedded in ``case`` and never regenerates test
    samples. W1/W2 materialize their fixed test slice into the case output Store
    so every metric has exact stateful Dataset authority. All metrics are
    checkpoint-bound Artifacts.
    """

    from dryml.artifacts import CachedDataset
    from dryml.core import Par
    from dryml.models import Experiment

    if case.accelerator == "gpu":
        initialize_gpu_framework(case.framework, case.seed["framework"])
    else:
        initialize_cpu_framework(case.framework, case.seed["framework"])
    model, trainer = _model_and_training(
        case.framework, case.workload, seed=case.seed["framework"],
    )
    if case.workload in {"W1", "W2"}:
        if mnist_source is None:
            raise QualificationUnrun("W1/W2 require caller-prepared TFDS fixture sources.")
        label_dtype = "int32" if case.framework == "jax" and case.workload == "W1" else None
        train = Batch(
            mnist_pipeline(
                mnist_source("train[:4096]", case.tensorflow_mode),
                autoencode=case.workload == "W2",
                label_dtype=label_dtype,
            ),
            64,
        )
        test = mnist_pipeline(
            mnist_source("test[:1024]", case.tensorflow_mode),
            autoencode=case.workload == "W2",
            label_dtype=label_dtype,
        )
        test_ref = CachedDataset(test, repo=repo).compute(
            codec="numpy",
            store=repo.stores[0],
            managed=managed,
        )
    else:
        train = Batch(
            as_supervised(
                polynomial_samples(seed=case.seed["train"], count=4096),
                "x",
                "y",
            ),
            64,
        )
        test_ref = case.w3_test_ref
    if case.workload == "W1":
        artifact_name = "accuracy"
        artifact = Definition(
            _QualificationMetric,
            Par("this.test_data"),
            Par("this.model"),
            metric="accuracy",
            x=(0,),
            y=(1,),
            fixture_store=None,
        )
    else:
        artifact_name = _METRIC_NAMES[case.workload]
        x_path, y_path = ((0,), (1,)) if case.workload == "W2" else ("x", "y")
        artifact = Definition(
            _QualificationMetric,
            Par("this.test_data"),
            Par("this.model"),
            metric="mse",
            x=x_path,
            y=y_path,
            fixture_store=None if case.workload == "W2" else case.fixture_store,
        )
    return Experiment(
        model, trainer, train_data=train, test_data=test_ref,
        artifacts={artifact_name: artifact}, checkpoint_every_steps=32,
    )


__all__ = [
    "CASE_KINDS", "EXECUTION_MODES", "FRAMEWORKS", "QualificationCase", "QualificationCasePaths", "QualificationEvidence", "THRESHOLDS", "WORKLOADS",
    "accelerated_cases", "accuracy_formula", "build_w3_fixtures", "build_workload", "case_from_manifest", "cpu_matrix",
    "initialize_cpu_framework", "initialize_gpu_framework", "mnist_pipeline", "mse_formula", "native_device_evidence", "native_device_observations", "observe_jax_training_tensors", "polynomial_samples", "preflight_gpu_framework", "prepare_exact_model_for_formula", "qualification_case_paths", "require_real_qualification", "run_local_case", "selected_gpu_device",
    "supplemental_tfds_torch_case", "validate_evidence", "w1_label_methods",
]

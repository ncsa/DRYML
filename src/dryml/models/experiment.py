from __future__ import annotations

import inspect
from types import SimpleNamespace

from dryml.artifacts import CachedDataset, NUMPY_SEQUENCE_KIND, PARQUET_KIND
from dryml.core import RefCDef, Repo
from dryml.core.definition import ConcreteDefinition
from dryml.core.object import Object, Serializable
from dryml.core.repo import get_default_repo
from dryml.core.symbol import resolve_symbol
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    DelegatedOutputs,
    ManagedCapabilityError,
    ManagedMethod,
    ManagedOutputRef,
    OperationPreflight,
    OperationResult,
    current_operation_context,
    managed,
    resolve_managed_store,
)
from dryml.records import (
    DataRecord,
    ExecutionRecordLink,
    RepresentationSpec,
    StoredStateRecord,
)

from .train_func import TrainFunction
from .train_spec import (
    TRAIN_CHECKPOINT_SCHEMA,
    TrainResumeMode,
    TrainState,
)
from .utils import hydrate_model_state, snapshot_model_state, write_model_state_output


class Experiment(Serializable):
    """Logical training recipe whose heavy results are managed realizations.

    Model, trainer, and dataset constructor arguments are non-materializing
    definition edges. ``model_state`` may select a prior Experiment's logical
    train output for fine-tuning; otherwise training snapshots the selected
    Store's ordinary model state before execution.
    """

    def __init__(
        self,
        model: RefCDef,
        train_fn: RefCDef,
        *,
        train_data: RefCDef | None = None,
        val_data: RefCDef | None = None,
        test_data: RefCDef | None = None,
        model_state: ManagedOutputRef | None = None,
        metrics=None,
        **capabilities,
    ):
        super().__init__()
        if not isinstance(model, ConcreteDefinition):
            raise TypeError("Experiment model must resolve to a ConcreteDefinition")
        if not isinstance(train_fn, ConcreteDefinition):
            raise TypeError("Experiment train_fn must resolve to a ConcreteDefinition")
        for name, value in (
            ("train_data", train_data),
            ("val_data", val_data),
            ("test_data", test_data),
        ):
            if value is not None and not isinstance(value, ConcreteDefinition):
                raise TypeError(f"Experiment {name} must resolve to a ConcreteDefinition")
        if model_state is not None and not isinstance(model_state, ManagedOutputRef):
            raise TypeError("Experiment model_state must be a ManagedOutputRef or None")

        self.model_definition = model
        self.train_fn_definition = train_fn
        self.train_data_definition = train_data
        self.val_data_definition = val_data
        self.test_data_definition = test_data
        self.model_state = model_state
        self.metrics = dict(metrics or {})
        self.capabilities = dict(capabilities)
        self.state = TrainState()

        # Existing direct-mode callers retain concrete compatibility when their
        # constructor objects are still available in the construction Repo.
        self.model = _cached_object(model)
        self.train_fn = _cached_object(train_fn)
        self.train_data = _cached_object(train_data)
        self.val_data = _cached_object(val_data)
        self.test_data = _cached_object(test_data)

    def __dryml_managed_preflight__(self, method, args, kwargs):
        if method != "train":
            raise ManagedCapabilityError(f"unsupported Experiment managed method {method!r}")
        if args or kwargs:
            raise TypeError("Experiment.train accepts only managed runtime arguments")
        trainer_cls = resolve_symbol(self.train_fn_definition.cls)
        if not issubclass(trainer_cls, TrainFunction):
            raise TypeError("Experiment train_fn must be a TrainFunction")
        capability = trainer_cls.resume_capability(self.train_fn_definition)
        pipeline_capability = getattr(
            trainer_cls,
            "managed_pipeline_capability",
            None,
        )
        if pipeline_capability is not None:
            pipeline_configuration = dict(self.capabilities)
            if self.metrics and pipeline_configuration.get("metrics") is None:
                pipeline_configuration["metrics"] = self.metrics
            capability = pipeline_capability(
                self.train_fn_definition,
                pipeline_configuration,
            )
        exact = capability.mode is TrainResumeMode.EXACT
        return OperationPreflight(
            resumable=exact,
            checkpoint_schema=capability.checkpoint_schema if exact else None,
            early_completion=capability.early_completion,
        )

    def __dryml_managed_inputs__(self, method, args, kwargs):
        if method != "train" or args or kwargs:
            raise TypeError("invalid Experiment train input request")
        refs = []
        for name, definition in self._data_definitions():
            if definition is None:
                continue
            refs.append(_cached_data_ref(definition, name))
        if self.model_state is not None:
            refs.append(self.model_state)
        return tuple(refs)

    def __dryml_managed_record_inputs__(self, method, args, kwargs, *, store):
        if method != "train" or args or kwargs:
            raise TypeError("invalid Experiment train record input request")
        if self.model_state is not None:
            return ()
        snapshot = snapshot_model_state(self.model_definition, store)
        record = StoredStateRecord.from_envelope(store.records.read_record(snapshot.record_id))
        return (
            ExecutionRecordLink(
                snapshot.record_id,
                role="initial-model-state",
                representation_id=record.representation_id,
                subject_cdef_id=record.subject_cdef_id,
            ),
        )

    def __dryml_managed_validate_inputs__(
        self,
        method,
        args,
        kwargs,
        *,
        store,
        consumed_records,
        consumed_record_links,
    ):
        """Reject incompatible exact cache or model-state records before work."""

        if method != "train" or args or kwargs:
            raise TypeError("invalid Experiment train input validation request")
        del consumed_record_links
        data_definitions = tuple(
            definition
            for _name, definition in self._data_definitions()
            if definition is not None
        )
        expected_count = len(data_definitions) + (self.model_state is not None)
        if len(consumed_records) != expected_count:
            raise ManagedCapabilityError("Experiment resolved input count is incompatible")
        for definition, resolved in zip(data_definitions, consumed_records):
            record = DataRecord.from_envelope(store.records.read_record(resolved.record_id))
            expected_subject = format_cdef_id(definition.stable_hash())
            if record.subject_cdef_id != expected_subject:
                raise ManagedCapabilityError(
                    "Experiment cache input subject does not match its definition"
                )
            representation = RepresentationSpec(
                store.records.read_spec(
                    record.representation_id,
                    family="representation",
                )
            )
            if representation.kind not in {NUMPY_SEQUENCE_KIND, PARQUET_KIND}:
                raise ManagedCapabilityError(
                    "Experiment cache input representation is unsupported"
                )
        if self.model_state is not None:
            model_record = StoredStateRecord.from_envelope(
                store.records.read_record(consumed_records[-1].record_id)
            )
            expected_subject = format_cdef_id(self.model_definition.stable_hash())
            if model_record.subject_cdef_id != expected_subject:
                raise ManagedCapabilityError(
                    "Experiment model_state subject does not match model definition"
                )

    @managed(
        outputs=DelegatedOutputs(("train_fn",)),
        resumable=True,
        checkpoint_schema=TRAIN_CHECKPOINT_SCHEMA,
        early_completion=True,
    )
    def train(self):
        """Train from exact completed caches and publish immutable model state.

        Without repository context, the descriptor retains the historical
        concrete in-memory behavior. Managed execution always trains a fresh
        model instance and returns a ``ManagedInvocationResult``.
        """

        try:
            context = current_operation_context()
        except RuntimeError:
            return self._train_direct()

        data_count = sum(definition is not None for _name, definition in self._data_definitions())
        data_records = context.consumed_records[:data_count]
        data_views = {}
        for (name, definition), resolved in zip(
            ((name, value) for name, value in self._data_definitions() if value is not None),
            data_records,
        ):
            dataset = Repo(context.store).load_or_build(
                definition,
                instance="new",
                cache="none",
                restore_state=False,
            )
            if not isinstance(dataset, CachedDataset):
                raise TypeError("Experiment training inputs must be CachedDataset definitions")
            data_views[name] = dataset.view_record(resolved.record_id, store=context.store)

        if self.model_state is not None:
            model_record_id = context.consumed_records[data_count].record_id
        else:
            if len(context.consumed_record_links) != 1:
                raise RuntimeError("Experiment training requires one initial model-state snapshot")
            model_record_id = context.consumed_record_links[0].record_id
        model = hydrate_model_state(self.model_definition, model_record_id, context.store)
        trainer = Repo(context.store).load_or_build(
            self.train_fn_definition,
            instance="new",
            cache="none",
            restore_state=False,
        )
        if not isinstance(trainer, TrainFunction):
            raise TypeError("Experiment train_fn did not materialize as a TrainFunction")

        runtime_exp = SimpleNamespace(
            model=model,
            train_fn=trainer,
            train_data=data_views.get("train_data"),
            val_data=data_views.get("val_data"),
            test_data=data_views.get("test_data"),
            metrics=dict(self.metrics),
            capabilities=dict(self.capabilities),
            state=TrainState(),
            resume_payload=None,
        )
        if context.is_resume:
            if context.checkpoint_path is None:
                raise RuntimeError("resumed Experiment has no committed checkpoint")
            runtime_exp.resume_payload = trainer.restore_checkpoint(
                runtime_exp, context.checkpoint_path
            )
        runtime_exp.state.phase = TrainState.training
        training_result = trainer(runtime_exp)
        if runtime_exp.state.phase == TrainState.training:
            runtime_exp.state.phase = TrainState.trained

        primary = context.outputs.primary.slot
        write_model_state_output(context, primary, runtime_exp.model)
        remaining = tuple(slot for slot in context.outputs.slots if slot != primary)
        if remaining:
            publisher = getattr(trainer, "publish_outputs", None)
            if publisher is None:
                raise RuntimeError(
                    "TrainFunction declared optional outputs without publish_outputs()"
                )
            publisher(runtime_exp, context, remaining)
        return training_result if isinstance(training_result, OperationResult) else None

    def trained_model(self, repo=None, *, store=None):
        """Hydrate the active train primary output into a fresh model instance."""

        selected = resolve_managed_store(repo, store=store, target=self)
        result = self.train.results(store=selected).get(self.train.result.slot)
        if result is None:
            raise RuntimeError("Experiment has no completed compatible trained model")
        return hydrate_model_state(self.model_definition, result.record_id, selected)

    def _train_direct(self):
        self.model = _materialize_concrete(self.model, self.model_definition)
        self.train_fn = _materialize_concrete(self.train_fn, self.train_fn_definition)
        self.train_data = _materialize_concrete(self.train_data, self.train_data_definition)
        self.val_data = _materialize_concrete(self.val_data, self.val_data_definition)
        self.test_data = _materialize_concrete(self.test_data, self.test_data_definition)
        self.state.phase = TrainState.training
        try:
            result = self.train_fn(self)
        except Exception:
            self.state.phase = TrainState.failed
            raise
        if self.state.phase == TrainState.training:
            self.state.phase = TrainState.trained
        return result

    def _data_definitions(self):
        return (
            ("train_data", self.train_data_definition),
            ("val_data", self.val_data_definition),
            ("test_data", self.test_data_definition),
        )


def _cached_data_ref(definition: ConcreteDefinition, name: str) -> ManagedOutputRef:
    cls = resolve_symbol(definition.cls)
    if not issubclass(cls, CachedDataset):
        raise TypeError(f"Experiment {name} must be a CachedDataset")
    descriptor = inspect.getattr_static(cls, "compute")
    if not isinstance(descriptor, ManagedMethod):
        raise TypeError(f"Experiment {name} has no managed compute output")
    return descriptor.output_ref(definition, "data")


def _cached_object(definition):
    if definition is None:
        return None
    repo = get_default_repo()
    if repo is None:
        return definition
    cached = repo.get_cached(definition)
    return definition if cached is None else cached


def _materialize_concrete(value, definition):
    if definition is None or isinstance(value, Object):
        return value
    repo = get_default_repo() or Repo()
    return repo.load_or_build(definition)


__all__ = ["Experiment"]

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dryml.core2 import Definition, Object, RefCDef
from dryml.managed import (
    DelegatedOutputs,
    DuplicateOutputError,
    InvalidSubjectPathError,
    ManagedOutput,
    ManagedOutputs,
    PrimaryOutputError,
    UnstableOutputsError,
    managed,
)


class Model(Object):
    pass


class StableTrainFunction(Object):
    __dryml_managed_outputs__ = ManagedOutputs(
        ManagedOutput("model", primary=True, kind="stored_state"),
        ManagedOutput("history", kind="data"),
    )


class AlternatingTrainFunction(Object):
    calls = 0

    @classmethod
    def __dryml_managed_outputs__(cls, definition):
        cls.calls += 1
        optional = "even" if cls.calls % 2 == 0 else "odd"
        return ManagedOutputs(
            ManagedOutput("model", primary=True),
            ManagedOutput(optional),
        )


class Experiment(Object):
    def __init__(self, model: RefCDef, train_fn: RefCDef):
        self.model = model
        self.train_fn = train_fn

    @managed(outputs=DelegatedOutputs(("kwargs", "train_fn")))
    def train(self):
        return "trained"


class SubjectOwner(Object):
    def __init__(self, model: RefCDef):
        self.model = model

    @managed(outputs=(ManagedOutput("result", primary=True, subject_path=("kwargs", "model")),))
    def compute(self):
        return None


class InvalidSubjectOwner(Object):
    @managed(outputs=(ManagedOutput("result", primary=True, subject_path=("kwargs", "missing")),))
    def compute(self):
        return None


def test_outputs_are_immutable_and_require_exactly_one_primary():
    declaration = ManagedOutput(
        "result",
        primary=True,
        kind="data",
        subject_path=("kwargs", "model"),
        representations=("numpy",),
    )
    outputs = ManagedOutputs(declaration, ManagedOutput("summary", kind="data"))

    assert outputs.primary is declaration
    assert outputs.slots == ("result", "summary")
    with pytest.raises(FrozenInstanceError):
        declaration.slot = "changed"
    with pytest.raises(PrimaryOutputError):
        ManagedOutputs(ManagedOutput("one"), ManagedOutput("two"))
    with pytest.raises(PrimaryOutputError):
        ManagedOutputs(ManagedOutput("one", primary=True), ManagedOutput("two", primary=True))
    with pytest.raises(DuplicateOutputError):
        ManagedOutputs(ManagedOutput("same", primary=True), ManagedOutput("same"))


@pytest.mark.parametrize("slot", ["", "not dotted", "a/b"])
def test_invalid_output_slots_are_rejected(slot):
    with pytest.raises(ValueError):
        ManagedOutput(slot, primary=True)


def test_delegated_outputs_are_derived_from_definition_without_materialization():
    model = Definition(Model).concretize()
    train_fn = Definition(StableTrainFunction).concretize()
    experiment = Definition(Experiment, model=model, train_fn=train_fn).concretize()

    declarations = Experiment.__dict__["train"].output_declarations(experiment)

    assert declarations.slots == ("model", "history")
    assert declarations.primary.slot == "model"


def test_unstable_delegated_output_contract_is_rejected():
    AlternatingTrainFunction.calls = 0
    experiment = Definition(
        Experiment,
        model=Definition(Model),
        train_fn=Definition(AlternatingTrainFunction),
    ).concretize()

    with pytest.raises(UnstableOutputsError):
        Experiment.__dict__["train"].output_declarations(experiment)


def test_subject_paths_are_validated_against_the_producer_definition():
    valid = Definition(SubjectOwner, model=Definition(Model)).concretize()
    SubjectOwner.__dict__["compute"].output_declarations(valid)

    with pytest.raises(InvalidSubjectPathError):
        InvalidSubjectOwner.__dict__["compute"].output_declarations(
            Definition(InvalidSubjectOwner).concretize()
        )

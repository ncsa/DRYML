"""Tests for shared requirement combination orchestration."""

import pytest

from dryml.requirements import (
    RequirementCombinationError,
    RequirementDeclaration,
    RequirementIssue,
    RequirementReport,
    RequirementResult,
    RequirementSource,
    combine_requirements,
)


class RecordingCombiner:
    """Minimal domain combiner used to observe shared orchestration input."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def combine(self, declarations):
        """Record declarations and return the configured domain outcome."""

        self.calls.append(declarations)
        return self.result if self.result is not None else RequirementResult("combined")


def _declaration(value, label):
    """Create one valid declaration for this module's tests."""

    return RequirementDeclaration(value, source=RequirementSource(label))


def test_empty_combination_succeeds_without_calling_the_domain_combiner():
    """No declarations return the sole legal empty result state."""

    combiner = RecordingCombiner()

    result = combine_requirements((), combiner=combiner)

    assert result == RequirementResult()
    assert combiner.calls == []


def test_combination_snapshots_a_one_shot_iterable_and_ordinalizes_sources():
    """Combination consumes ordered declarations once before semantic delegation."""

    combiner = RecordingCombiner()
    result = combine_requirements((_declaration(value, label) for value, label in ((1, "one"), (2, "two"))), combiner=combiner)

    assert result.value == "combined"
    assert [item.source.label for item in combiner.calls[0]] == ["1: one", "2: two"]


def test_combination_rejects_invalid_input_before_a_partial_result_is_exposed():
    """Malformed declarations and combiner outcomes fail closed."""

    combiner = RecordingCombiner()
    with pytest.raises(RequirementCombinationError):
        combine_requirements((_declaration(1, "one"), object()), combiner=combiner)
    assert combiner.calls == []

    malformed = RecordingCombiner(RequirementResult())
    with pytest.raises(RequirementCombinationError):
        combine_requirements((_declaration(1, "one"),), combiner=malformed)

    conflict = RecordingCombiner(RequirementResult(report=RequirementReport((RequirementIssue("example.conflict", "conflict"),))))
    assert combine_requirements((_declaration(1, "one"),), combiner=conflict).has_value is False

    with pytest.raises(RequirementCombinationError):
        combine_requirements((_declaration(1, "one"),), combiner=RecordingCombiner(object()))


def test_combination_enforces_the_declaration_boundary_with_limit_plus_one_consumption():
    """The shared declaration budget never reads beyond the rejecting item."""

    combiner = RecordingCombiner()
    accepted = tuple(_declaration(index, str(index)) for index in range(256))
    assert combine_requirements(accepted, combiner=combiner).has_value
    seen = []

    def overflow():
        for index in range(257):
            seen.append(index)
            yield _declaration(index, str(index))

    with pytest.raises(RequirementCombinationError):
        combine_requirements(overflow(), combiner=combiner)
    assert seen == list(range(257))


def test_combination_ordinalizes_a_maximum_length_source_without_overflowing_it():
    """Ordinal source context preserves the source bound at its largest input."""

    combiner = RecordingCombiner()
    combine_requirements((_declaration(1, "x" * 256),), combiner=combiner)

    assert combiner.calls[0][0].source.label.startswith("1: ")
    assert len(combiner.calls[0][0].source.label) == 256

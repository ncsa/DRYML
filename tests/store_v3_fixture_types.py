"""Stable author-payload types used by the checked-in Store v3 fixtures."""

from pathlib import Path

from dryml.core import Object, Serializable
from dryml.environments import req


class EmptyRequirementValue(Serializable):
    """Persist one small payload without an environment requirement."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        """Write the fixture value through the ordinary local-state codec hook."""

        Path(directory, "value.txt").write_text(str(self.value), encoding="ascii")


@req(python=">=3.11", requirements=("syntheticpkg>=2",), source="fixture-value")
class ValuedRequirementValue(Serializable):
    """Persist one payload with a satisfiable synthetic requirement."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        """Write the fixture value through the ordinary local-state codec hook."""

        Path(directory, "value.txt").write_text(str(self.value), encoding="ascii")


@req(requirements=("conflictpkg>=2",), source="fixture-conflict-new")
@req(requirements=("conflictpkg<1",), source="fixture-conflict-old")
class ConflictingRequirementValue(Serializable):
    """Persist one payload whose declarations produce a complete conflict."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        """Write the fixture value through the ordinary local-state codec hook."""

        Path(directory, "value.txt").write_text(str(self.value), encoding="ascii")


class RoutedStateRoot(Object):
    """Retain an exact StateRef child so the parent uses routed placement."""

    def __init__(self, child):
        self.child = child

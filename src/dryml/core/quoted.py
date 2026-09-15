from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QuotedDef:
    """Store a Definition expression as local constructor data, not an object edge.

    Args:
        value: Definition expression frozen into canonical selector data.

    ``Ref[QuotedDef]`` boundaries deliver this wrapper as expression data when a
    caller explicitly requests quotation rather than materialization.

    Raises:
        TypeError: If ``value`` cannot be represented as selector data.

    Side Effects:
        Freezes ``value`` during construction without selecting, materializing, or
        saving an Object.
    """

    value: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_selector_value

        object.__setattr__(self, "value", freeze_selector_value(self.value))


@dataclass(frozen=True, slots=True)
class SelectorSpec:
    """Store a Selector expression as local constructor data.

    Args:
        selector: Selector expression frozen into canonical selector data.

    ``Ref[SelectorSpec]`` boundaries deliver this wrapper as expression data when
    a caller explicitly requests selector quotation.

    Raises:
        TypeError: If ``selector`` cannot be represented as selector data.

    Side Effects:
        Freezes ``selector`` during construction without selecting,
        materializing, or saving an Object.
    """

    selector: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_selector_value

        object.__setattr__(self, "selector", freeze_selector_value(self.selector))

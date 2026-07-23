from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QuotedDef:
    """Store a Definition expression as local constructor data, not an object edge.

    SelectorArg constructors receive this wrapper at runtime when the stored
    value originated from a quoted Definition.
    """

    value: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_selector_value

        object.__setattr__(self, "value", freeze_selector_value(self.value))


@dataclass(frozen=True, slots=True)
class SelectorSpec:
    """Store a Selector expression as local constructor data.

    SelectorArg constructors receive this wrapper at runtime when the stored
    value originated from a Selector.
    """

    selector: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_selector_value

        object.__setattr__(self, "selector", freeze_selector_value(self.selector))

"""Canonical rendering for internal world diagnostic paths."""

from __future__ import annotations

import json
import re

WorldPath = tuple[str | int, ...]

_PLAIN_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


def path(*segments: str | int) -> WorldPath:
    """Return one structured path whose segment boundaries remain unambiguous."""

    return segments


def render_path(value: WorldPath) -> str:
    """Render a structured path with ordinary dotted spelling where safe."""

    rendered: list[str] = []
    for index, segment in enumerate(value):
        if type(segment) is int:
            rendered.append(f"[{segment}]")
        elif _PLAIN_PATH_SEGMENT.fullmatch(segment):
            rendered.append(segment if index == 0 else f".{segment}")
        else:
            rendered.append(f"[{json.dumps(segment, ensure_ascii=True)}]")
    return "".join(rendered)

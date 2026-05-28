import inspect
import textwrap
from dataclasses import dataclass

@dataclass(frozen=True)
class SourceInfo:
    """
    A type to hold the source code for a python object
    """
    source: str
    filename: str | None
    start_line: int | None


def get_source_info(obj) -> SourceInfo | None:
    """
    Method to get source code from a python object
    """
    try:
        lines, start_line = inspect.getsourcelines(obj)
        filename = inspect.getsourcefile(obj)
    except (OSError, TypeError):
        return None

    return SourceInfo(
        source=textwrap.dedent("".join(lines)),
        filename=filename,
        start_line=start_line,
    )


def func_source_extract(func):
    # Get source code for a given function,
    # and format it in a consistent way for
    # building custom transformations.
    #
    # Args:
    #   func: The function whose source to extract.

    # Get the source code
    lines, _ = inspect.getsourcelines(func)
    return textwrap.dedent("".join(lines))

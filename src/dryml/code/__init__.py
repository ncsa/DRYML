"""Generic dependency-light code analysis and source-transformation utilities.

This package deliberately does not expose Method declaration, trait, selection,
or preparation policy; those public APIs belong to :mod:`dryml.methods`.
"""

from .callable_info import CallableInfo, analyze_callable
from .source import SourceInfo, func_source_extract, get_source_info

__all__ = [
    "CallableInfo",
    "analyze_callable",
    "SourceInfo",
    "get_source_info",
    "func_source_extract",
]

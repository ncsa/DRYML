"""Small importable code-probe targets.

The functions below are safe to import and are not intended to be executed by
the probe service.
"""

from __future__ import annotations

import dryml
from dryml.core2.object import Object


BODY_EXECUTED = False


@dryml.env.req(requirements=("probepkg>=1",))
def decorated_function(value=1):
    return value + 1


def plain_function(value=1):
    return value + 1


@dryml.env.req(python=">=3")
def current_python_required_function():
    return None


class BenchmarkObject(Object):
    """Small persisted subject used by dispatch performance measurements."""

    def ping(self):
        return 1


class BenchmarkTraceModel:
    def ping(self):
        raise AssertionError("dynamic tracing must not invoke receiver methods")


def trace_zero():
    return None


def trace_one(model):
    model.ping()


def trace_repeated(model):
    for _ in range(16):
        model.ping()


def body_must_not_run():
    global BODY_EXECUTED
    BODY_EXECUTED = True
    raise AssertionError("probe executed function body")


class ProbeMethods:
    @dryml.env.req(requirements=("methodpkg>=1",))
    def train(self):
        return "should not run during probe"

    @classmethod
    @dryml.env.req(requirements=("classpkg>=1",))
    def build(cls):
        return cls()

    @staticmethod
    @dryml.env.req(requirements=("staticpkg>=1",))
    def make():
        return "should not run during probe"

    def __init__(self):
        raise AssertionError("probe instantiated ProbeMethods")

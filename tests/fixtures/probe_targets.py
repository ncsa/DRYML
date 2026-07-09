"""Small importable code-probe targets.

The functions below are safe to import and are not intended to be executed by
the probe service.
"""

from __future__ import annotations

import dryml


BODY_EXECUTED = False


@dryml.env.req(requirements=("probepkg>=1",))
def decorated_function(value=1):
    return value + 1


def plain_function(value=1):
    return value + 1


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

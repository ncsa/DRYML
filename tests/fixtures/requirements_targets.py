"""Reusable decorated targets for requirements/dispatch/code-analysis tests.

These fixtures intentionally avoid importing heavy frameworks. Package names in
requirements are metadata strings only; tests inspect annotation fragments rather
than importing the named packages.
"""

from __future__ import annotations

import dryml


@dryml.env.req(requirements=("torch>=2",))
class BaseTorchModel:
    @dryml.env.req(requirements=("numpy>=1.26",))
    def inherited_train(self, data=None):
        return {"target": "base.inherited_train", "data": data}

    def train(self, data=None):
        return {"target": "base.train", "data": data}


@dryml.env.req(requirements=("lightning>=2",))
class LightningModel(BaseTorchModel):
    @dryml.world.req(accelerators={"gpu": {"min": 1}})
    def train(self, data=None):
        return {"target": "lightning.train", "data": data}


@dryml.env.req(requirements=("pandas>=2",))
def run_training(experiment):
    experiment.train(None)
    return "trained"


class ClassMethodTargets:
    @classmethod
    @dryml.env.req(requirements=("inner-classmethod>=1",))
    def inner_decorated(cls):
        return cls.__name__

    @dryml.env.req(requirements=("outer-classmethod>=1",))
    @classmethod
    def outer_decorated(cls):
        return cls.__name__


class StaticMethodTargets:
    @staticmethod
    @dryml.env.req(requirements=("inner-staticmethod>=1",))
    def inner_decorated():
        return "inner-static"

    @dryml.env.req(requirements=("outer-staticmethod>=1",))
    @staticmethod
    def outer_decorated():
        return "outer-static"


def plain_importable_function(value=1):
    return value + 1


def make_local_training_function():
    @dryml.env.req(requirements=("local-only>=1",))
    def local_training(experiment):
        experiment.train(None)
        return "local-trained"

    return local_training


local_lambda_with_annotation = dryml.env.req(requirements=("lambda-only>=1",))(
    lambda experiment: experiment.train(None)
)

"""Inert CPU XGBoost Model recipes and one-shot Dataset training.

Importing this namespace does not import XGBoost or sklearn. Native dependencies
are admitted at fit/fitted restore; see docs/xgboost.md for the supported stack.
"""

from dryml.models.xgb.base import BasicTraining, Model, ClassifierModel, RegressionModel

__all__ = [
    "Model",
    "ClassifierModel",
    "RegressionModel",
    "BasicTraining",
]

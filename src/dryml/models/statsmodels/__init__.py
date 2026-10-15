"""Passive OLS/binary Logit recipes and one-shot Dataset training.

Imports require no statsmodels, sklearn or XGBoost. Native fitting and fitted
restoration admit the version-bound stack documented in ``docs/statsmodels.md``.
"""

from dryml.models.statsmodels.base import BasicTraining, Model, ClassifierModel, RegressionModel

__all__ = ["Model", "ClassifierModel", "RegressionModel", "BasicTraining"]

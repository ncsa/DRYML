"""Eager scalar metrics and inert native streaming metric Fold factories."""

from dryml.metrics.scalar import categorical_accuracy, mean_squared_error
from dryml.metrics.reductions import (
    AccuracyFromConfusion,
    ConfusionCounts,
    ConfusionInitial,
    F1Average,
    F1FromConfusion,
    Label,
    classifier_accuracy,
    classifier_confusion_matrix,
    classifier_f1,
    regressor_mae,
    regressor_mse,
)

__all__ = [
    "mean_squared_error",
    "categorical_accuracy",
    "AccuracyFromConfusion",
    "ConfusionCounts",
    "ConfusionInitial",
    "F1Average",
    "F1FromConfusion",
    "Label",
    "classifier_accuracy",
    "classifier_confusion_matrix",
    "classifier_f1",
    "regressor_mae",
    "regressor_mse",
]

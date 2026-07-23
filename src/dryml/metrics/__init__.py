from dryml.metrics.scalar import categorical_accuracy, mean_squared_error

__all__ = [
    "CategoricalAccuracy",
    "ConfusionMatrix",
    "mean_squared_error",
    "categorical_accuracy",
]


def __getattr__(name):
    if name in {"CategoricalAccuracy", "ConfusionMatrix"}:
        from dryml.metrics import classification

        value = getattr(classification, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dryml.metrics' has no attribute {name!r}")

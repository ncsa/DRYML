__all__ = [
    "AutoEncoder",
    "Experiment",
    "ExperimentData",
    "ExperimentDataError",
    "Model",
    "TrainFunction",
    "TrainState",
]


def __getattr__(name):
    """Load model APIs only when their public name is requested.

    Args:
        name: Public attribute requested from ``dryml.models``.

    Returns:
        The lazily imported public model API.

    Raises:
        AttributeError: If ``name`` is not a public model API.

    Side Effects:
        Imports only the module that owns the requested API and caches the value
        in this module. Requesting ``ExperimentData`` does not import pandas.

    This keeps ExperimentData's required pandas dependency and optional framework
    backends out of lightweight model/reference discovery imports.
    """

    modules = {
        "Experiment": (".experiment", "Experiment"),
        "ExperimentData": (".experiment_data", "ExperimentData"),
        "ExperimentDataError": (".experiment_data", "ExperimentDataError"),
        "AutoEncoder": (".model", "AutoEncoder"),
        "Model": (".model", "Model"),
        "TrainFunction": (".train_func", "TrainFunction"),
        "TrainState": (".train_spec", "TrainState"),
    }
    try:
        module_name, attribute = modules[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

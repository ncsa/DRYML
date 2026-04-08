from fixtures import store_resource_factory, create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, primary_store_set, ray
import pytest

from mk_ic import install
from mk_ic import pytest_wrapper_elimination as _pwe
install()
ics.configureOutput(frame_filters=[_pwe])

from dryml.context.context_tracker import add_context, ContextError

def pytest_sessionstart(session):
    # import jax needs to go before tensorflow
    # Enforce special loading order to prevent crash
    # https://github.com/pytorch/pytorch/issues/101152
    #import torch  # noqa: F401
    #import tensorflow as tf  # noqa: F401

    for ctx_name in ("jax", "torch", "tf"):
        try:
            add_context(ctx_name)
        except ContextError as e:
            pass


__all__ = [
    store_resource_factory,
    primary_store_set,
    create_name,
    create_temp_file,
    create_temp_dir,
    create_temp_named_file,
    ray,
]

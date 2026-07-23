from fixtures import store_resource_factory, create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, primary_store_set, ray, \
    sample_environment_record
import builtins
import os
import sys

_CORE_HELPERS = os.path.join(os.path.dirname(__file__), "core")
if _CORE_HELPERS not in sys.path:
    sys.path.insert(0, _CORE_HELPERS)

pytest_plugins = ("timing_plugin",)

try:
    from mk_ic import install
    from mk_ic import pytest_wrapper_elimination as _pwe
except ImportError:
    builtins.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
else:
    install()
    ics.configureOutput(frame_filters=[_pwe])

__all__ = [
    store_resource_factory,
    primary_store_set,
    create_name,
    create_temp_file,
    create_temp_dir,
    create_temp_named_file,
    ray,
    sample_environment_record,
]

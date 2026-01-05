from fixtures import store_resource_factory, create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, primary_store_set, ray

from mk_ic import install
from mk_ic import pytest_wrapper_elimination as _pwe
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
]

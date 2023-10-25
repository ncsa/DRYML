from fixtures import create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, ray_server, \
    get_ray

from mk_ic import install
from mk_ic import pytest_wrapper_elimination as _pwe
install()
ics.configureOutput(frame_filters=[_pwe])

__all__ = [
    create_name,
    create_temp_file,
    create_temp_dir,
    create_temp_named_file,
    ray_server,
    get_ray,
]

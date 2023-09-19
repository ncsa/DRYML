import pytest
import tempfile
import uuid
import os


if os.environ.get('GITHUB_ACTIONS') == 'true':
    # Enforce special loading order to prevent crash
    # https://github.com/pytorch/pytorch/issues/101152
    import torch  # noqa: F401
    import tensorflow as tf  # noqa: F401


@pytest.fixture
def create_name():
    tempf = str(uuid.uuid4())
    yield tempf
    fullpath = f"{tempf}.dry"
    if os.path.exists(fullpath):
        os.remove(fullpath)
    if os.path.exists(tempf):
        os.remove(tempf)


@pytest.fixture
def create_temp_named_file():
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        yield f.name


@pytest.fixture
def create_temp_file():
    # We need to open with 'w+b' permission so that we can both
    # Read and write
    with tempfile.TemporaryFile(mode='w+b') as f:
        yield f


@pytest.fixture
def create_temp_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory


@pytest.fixture
def get_ray():
    try:
        import ray
    except ImportError:
        pytest.skip("Ray is not installed")
    else:
        yield ray


@pytest.fixture(scope="session", autouse=True)
def ray_server(request):
    try:
        import ray
        ray.init(num_cpus=1, num_gpus=0)

        def shutdown_ray():
            ray.shutdown()
        request.addfinalizer(shutdown_ray)
    except ImportError:
        pass

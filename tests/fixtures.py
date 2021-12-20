import pytest
import tempfile
import uuid
import os


@pytest.fixture
def create_name():
    tempf = str(uuid.uuid4())
    yield tempf
    fullpath = f"{tempf}.dry"
    if os.path.exists(fullpath):
        os.remove(fullpath)


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

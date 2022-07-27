from dryml.file_intermediary import FileWriteIntermediary
import zipfile
import pytest
import pickle
from dryml.utils import pickler


@pytest.mark.usefixtures("create_name")
def test_file_intermediary_1(create_name):
    # Create intermediary
    int_file = FileWriteIntermediary()

    test_file = 'test.txt'
    test_text = "TEST"

    # Create test zip as usual
    z_file = zipfile.ZipFile(int_file, mode='w')
    with z_file.open("test.txt", 'w') as f:
        f.write(test_text.encode('utf-8'))
    z_file.close()

    # Write to the temp location
    with open(create_name, 'wb') as f:
        int_file.write_to_file(f)
    int_file.close()

    # Read temp file into zipfile.
    with open(create_name, 'rb') as f:
        z_file = zipfile.ZipFile(f, mode='r')
        assert test_file in z_file.namelist()
        with z_file.open(test_file, 'r') as f2:
            f2.read().decode('utf-8') == test_text


@pytest.mark.usefixtures("create_name")
def test_file_intermediary_2(create_name):
    # Create intermediary
    int_file = FileWriteIntermediary()

    test_file = 'test.txt'
    test_text = "TEST"

    # Create test zip as usual
    z_file = zipfile.ZipFile(int_file, mode='w')
    with z_file.open("test.txt", 'w') as f:
        f.write(test_text.encode('utf-8'))
    z_file.close()

    # Write to the temp location and close intermediary
    int_file.write_to_file(create_name)
    int_file.close()

    # Read temp file into zipfile.
    with open(create_name, 'rb') as f:
        z_file = zipfile.ZipFile(f, mode='r')
        assert test_file in z_file.namelist()
        with z_file.open(test_file, 'r') as f2:
            f2.read().decode('utf-8') == test_text


@pytest.mark.usefixtures("create_name")
def test_file_intermediary_3(create_name):
    # Create intermediary
    int_file = FileWriteIntermediary(mem_mode=True)

    test_file = 'test.txt'
    test_text = "TEST"

    # Create test zip as usual
    z_file = zipfile.ZipFile(int_file, mode='w')
    with z_file.open("test.txt", 'w') as f:
        f.write(test_text.encode('utf-8'))
    z_file.close()

    # Write to the temp location
    with open(create_name, 'wb') as f:
        int_file.write_to_file(f)
    int_file.close()

    # Read temp file into zipfile.
    with open(create_name, 'rb') as f:
        z_file = zipfile.ZipFile(f, mode='r')
        assert test_file in z_file.namelist()
        with z_file.open(test_file, 'r') as f2:
            f2.read().decode('utf-8') == test_text


@pytest.mark.usefixtures("create_name")
def test_file_intermediary_4(create_name):
    # Create intermediary
    int_file = FileWriteIntermediary(mem_mode=True)

    test_file = 'test.txt'
    test_text = "TEST"

    # Create test zip as usual
    z_file = zipfile.ZipFile(int_file, mode='w')
    with z_file.open("test.txt", 'w') as f:
        f.write(test_text.encode('utf-8'))
    z_file.close()

    # Write to the temp location and close intermediary
    int_file.write_to_file(create_name)
    int_file.close()

    # Read temp file into zipfile.
    with open(create_name, 'rb') as f:
        z_file = zipfile.ZipFile(f, mode='r')
        assert test_file in z_file.namelist()
        with z_file.open(test_file, 'r') as f2:
            f2.read().decode('utf-8') == test_text


@pytest.mark.usefixtures("create_name")
def test_file_intermediary_5(create_name):
    int_file = FileWriteIntermediary()
    z_file = zipfile.ZipFile(int_file, mode='w')

    # Save meta data analogue
    meta_data = {
        'version': 1
    }

    meta_dump = pickler(meta_data)
    with z_file.open('meta_data.pkl', mode='w') as f:
        f.write(meta_dump)

    z_file.close()
    int_file.write_to_file(create_name)
    int_file.close()

    # Load meta data analogue
    with open(create_name, 'rb') as file:
        z_file = zipfile.ZipFile(file, mode='r')

        with z_file.open('meta_data.pkl', 'r') as meta_file:
            assert pickle.loads(meta_file.read()) == meta_data

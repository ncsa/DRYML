import os
from pathlib import Path
import subprocess
import sys


def test_formats_and_environment_base_imports_do_not_load_probe_or_frameworks():
    source = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ | {"PYTHONPATH": os.pathsep.join((str(source), os.environ.get("PYTHONPATH", "")))}
    subprocess.run(
        [sys.executable, "-c", "import sys; import dryml.formats, dryml.environments; assert 'dryml.environments.probe' not in sys.modules; assert 'dryml.environments.probe_worker' not in sys.modules; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules"],
        check=True,
        env=environment,
    )

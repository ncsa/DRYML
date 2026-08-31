from pathlib import Path
import inspect

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.dir import DirStore


class CodecBase(Serializable):
    calls = []

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).calls.append(("base", codec))
        Path(dest_dir, "base.txt").write_text(codec)


class CodecLeaf(CodecBase):
    def __init__(self, value=1):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).calls.append(("leaf", codec))
        Path(dest_dir, "leaf.txt").write_text(str(self.value))


@pytest.mark.parametrize("codec", ["pkl", "HDF5", "torch2"])
def test_codec_reaches_every_mro_hook_unchanged(tmp_path, codec):
    CodecLeaf.calls = []
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = CodecLeaf(repo=repo)
    obj.state_codec = codec

    state = obj.save(repo=repo)

    assert CodecLeaf.calls == [("leaf", codec), ("base", codec)]
    assert state.states[next(iter(state.states))].startswith(f"{codec}-")


def test_invalid_codec_fails_before_hook_or_store_mutation(tmp_path):
    CodecLeaf.calls = []
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = CodecLeaf(repo=repo)
    obj.state_codec = "not-valid!"

    with pytest.raises(Exception, match="state_codec"):
        obj.save(repo=repo)

    assert CodecLeaf.calls == []
    assert not (tmp_path / "store" / "local-state").exists()


def test_lightweight_backend_hooks_accept_keyword_only_codec_without_runtime_imports():
    from dryml.artifacts import Scalar
    from dryml.models.experiment import Experiment
    from dryml.models.tf.base import Model as TensorFlowModel
    from dryml.models.torch.base import Model as TorchModel

    for hook in (
            Scalar.save_state_to_dir_imp,
            Experiment.save_state_to_dir_imp,
            TensorFlowModel.save_state_to_dir_imp,
            TorchModel.save_state_to_dir_imp):
        parameter = inspect.signature(hook).parameters["codec"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

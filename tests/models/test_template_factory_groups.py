"""Build-only qualification for resolved template factory groups."""

from __future__ import annotations

import pytest

from dryml.core import F, Par, Shared, Template, TemplateGenerator, UniformFromSet


def _resolved_groups(sequential, factories):
    """Generate one resolved Sequential Definition without loading data."""

    template = Template(
        sequential,
        layer_defs=factories * Par("depth"),
    )
    return TemplateGenerator(
        template,
        depth=UniformFromSet((2,)),
        width=4,
    ).sample()


def test_tf_template_factory_groups_build_with_resolved_values_only():
    """TensorFlow factories build from a completed Definition without training."""

    pytest.importorskip("tensorflow")
    from dryml.models.tf import Sequential

    definition = _resolved_groups(
        Sequential,
        [F("Dense", units=Par("width")), F("ReLU")],
    )
    model = definition.cls(*definition.args, **definition.kwargs)

    assert len(model.obj.layers) == 4
    assert model.obj.layers[0] is not model.obj.layers[2]


def test_torch_template_factory_groups_build_with_resolved_values_only():
    """PyTorch factories build from a completed Definition without training."""

    torch = pytest.importorskip("torch")
    from dryml.models.torch import Sequential

    definition = _resolved_groups(
        Sequential,
        [F("Linear", 3, Par("width")), F("ReLU")],
    )
    model = definition.cls(*definition.args, **definition.kwargs)

    assert len(model.obj) == 4
    assert model.obj[0] is not model.obj[2]
    assert isinstance(model.obj[0], torch.nn.Linear)


def test_shared_factory_repetition_does_not_tie_backend_instances():
    """Shared template topology does not promise backend factory instance reuse."""

    torch = pytest.importorskip("torch")
    from dryml.models.torch import Sequential

    template = Template(
        Sequential,
        layer_defs=[F("Linear", 3, 4)] * Shared(Par("depth")),
    )
    definition = TemplateGenerator(
        template, depth=UniformFromSet((2,))
    ).sample()
    model = definition.cls(*definition.args, **definition.kwargs)

    assert len(model.obj) == 2
    assert model.obj[0] is not model.obj[1]
    assert isinstance(model.obj[0], torch.nn.Linear)

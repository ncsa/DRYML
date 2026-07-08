from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import dryml
from dryml.annotations import fragments_for, fragments_for_class, fragments_for_callable


def _load_targets():
    path = Path(__file__).parents[1] / "fixtures" / "requirements_targets.py"
    spec = importlib.util.spec_from_file_location("dryml_requirement_targets", path)
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


targets = _load_targets()


def _requirements(fragments):
    reqs = []
    for fragment in fragments:
        reqs.extend(fragment.fragment.get("requirements", ()))
    return tuple(reqs)


def test_env_req_attaches_fragment_to_function_without_wrapping():
    def func():
        return "ok"

    decorated = dryml.env.req(requirements=("example>=1",))(func)

    assert decorated is func
    assert decorated() == "ok"
    fragments = fragments_for(decorated)
    assert len(fragments) == 1
    assert fragments[0].namespace == "environment"
    assert fragments[0].kind == "requirement"
    assert "example>=1" in fragments[0].fragment["requirements"]


def test_env_req_attaches_fragment_to_class_without_wrapping():
    class Model:
        pass

    decorated = dryml.env.req(requirements=("class-only>=1",))(Model)

    assert decorated is Model
    assert isinstance(decorated(), Model)
    assert "class-only>=1" in _requirements(fragments_for_class(decorated))


def test_world_req_attaches_fragment_to_method_without_wrapping():
    method = targets.LightningModel.__dict__["train"]
    fragments = fragments_for(method, namespace="world")

    assert targets.LightningModel().train(None)["target"] == "lightning.train"
    assert len(fragments) == 1
    assert fragments[0].fragment["roles"]["main"]["resources"]["accelerators"]["gpu"]["min"] == 1


def test_class_mro_fragments_include_current_base_then_subclass_behavior():
    fragments = fragments_for_class(targets.LightningModel, namespace="environment")

    assert _requirements(fragments) == ("torch>=2", "lightning>=2")


def test_callable_collection_current_method_owner_behavior():
    fragments = fragments_for_callable(targets.LightningModel().train)


    assert "torch>=2" in _requirements(fragments)
    assert "lightning>=2" in _requirements(fragments)
    assert any(fragment.namespace == "world" for fragment in fragments)


def test_inherited_method_fragment_current_behavior():
    fragments = fragments_for_callable(targets.LightningModel().inherited_train, namespace="environment")

    assert _requirements(fragments) == ("torch>=2", "lightning>=2", "numpy>=1.26")


def test_decorated_plain_function_has_annotation_fragment():
    assert targets.run_training(targets.LightningModel()) == "trained"
    assert _requirements(fragments_for(targets.run_training, namespace="environment")) == ("pandas>=2",)


def test_classmethod_inner_decorator_order_current_behavior():
    fragments = fragments_for_callable(targets.ClassMethodTargets.inner_decorated, namespace="environment")
    assert targets.ClassMethodTargets.inner_decorated() == "ClassMethodTargets"
    assert _requirements(fragments) == ("inner-classmethod>=1",)


def test_classmethod_outer_decorator_order_current_limitation():
    fragments = fragments_for_callable(targets.ClassMethodTargets.outer_decorated, namespace="environment")
    raw_fragments = fragments_for(targets.ClassMethodTargets.__dict__["outer_decorated"], namespace="environment")

    assert targets.ClassMethodTargets.outer_decorated() == "ClassMethodTargets"
    assert fragments == ()
    assert _requirements(raw_fragments) == ("outer-classmethod>=1",)


def test_staticmethod_inner_decorator_order_current_behavior():
    fragments = fragments_for_callable(targets.StaticMethodTargets.inner_decorated, namespace="environment")
    assert targets.StaticMethodTargets.inner_decorated() == "inner-static"
    assert _requirements(fragments) == ("inner-staticmethod>=1",)


def test_staticmethod_outer_decorator_order_current_limitation():
    fragments = fragments_for_callable(targets.StaticMethodTargets.outer_decorated, namespace="environment")
    raw_fragments = fragments_for(targets.StaticMethodTargets.__dict__["outer_decorated"], namespace="environment")

    assert targets.StaticMethodTargets.outer_decorated() == "outer-static"
    assert fragments == ()
    assert _requirements(raw_fragments) == ("outer-staticmethod>=1",)

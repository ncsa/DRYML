import dryml.annotations as ann
import dryml.env
import dryml.environments as envs


def test_env_sugar_resolves_to_environment_requirement():
    @dryml.env.req(packages={"torch": ">=2.4", "numpy": None}, requirements=("transformers>=4",), tags=("training",))
    def train():
        pass

    req = ann.resolve_environment_requirement(train)
    assert req.requirements == ("numpy", "torch>=2.4", "transformers>=4")
    assert req.tags == ("training",)


def test_class_hierarchy_and_legacy_env_decorator_collection():
    @envs.req(requirements=("dryml",))
    class Base:
        pass

    @dryml.env.req(packages={"torch": None})
    class Child(Base):
        pass

    fragments = ann.fragments_for_class(Child, namespace="environment")
    assert len(fragments) == 2
    assert ann.resolve_environment_requirement(Child).requirements == ("dryml", "torch")


def test_legacy_environment_override_matches_generic_resolution():
    @envs.req(requirements=("torch>=2",), tags=("torch",), python=">=3.10")
    class Base:
        pass

    @envs.override_req(requirements=("torch>=2.6",), python=">=3.11")
    class Child(Base):
        pass

    generic = ann.resolve_environment_requirement(Child)
    legacy = envs.requirements_for_class(Child)
    assert generic.requirements == legacy.requirements
    assert generic.tags == legacy.tags
    assert generic.python == legacy.python

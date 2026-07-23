import dryml
import dryml.annotations as ann


def test_public_sugar_and_low_level_equivalence():
    @dryml.env.req(packages={"torch": None})
    @dryml.world.req(cpus={"min": 1})
    @dryml.world.default(cpus=2)
    @dryml.runtime.default(torch={"num_threads": 2})
    def sugar():
        pass

    @ann.require(namespace="environment", fragment=dryml.env.normalize_environment_requirement_fragment(packages={"torch": None}))
    def low_level():
        pass

    assert ann.resolve(sugar).report.ok
    sugar_req = ann.resolve_environment_requirement(sugar)
    low_level_req = ann.resolve_environment_requirement(low_level)
    assert sugar_req.requirements == low_level_req.requirements


def test_arg_role_reexport_preserves_old_behavior():
    from dryml.annotations import RefCDef as NewRefCDef
    from dryml.core.arg_roles import RefCDef as OldRefCDef

    assert NewRefCDef == OldRefCDef

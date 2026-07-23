import dryml.annotations as ann
import dryml.world
from dryml.worlds import WorldRequirement, WorldSpec


def test_world_requirement_and_default_resolution():
    @dryml.world.req(cpus={"min": 2}, memory={"min": "1GiB"}, accelerators={"gpu": {"min": 1}})
    @dryml.world.default(cpus=4, memory="2GiB", accelerators={"gpu": 1})
    def train():
        pass

    assert isinstance(ann.resolve_world_requirement(train), WorldRequirement)
    assert isinstance(ann.resolve_world_default(train), WorldSpec)
    assert ann.resolve(train).report.ok


def test_class_method_world_merge_and_user_override_violation():
    @dryml.world.req(accelerators={"gpu": {"min": 1}})
    class Trainer:
        @dryml.world.default(cpus=8, accelerators={"gpu": 1})
        def run(self):
            pass

    override = {"world": {"roles": {"main": {"process": {"resources": {"accelerators": {"gpu": 0}}}}}}}
    result = ann.resolve(Trainer().run, overrides=override)
    assert not result.report.ok
    assert result.report.issues[0].sources


def test_world_default_convenience_accepts_direct_override_payload():
    @dryml.world.default(cpus=2)
    def train():
        pass

    world = ann.resolve_world_default(train, overrides={"roles": {"main": {"process": {"resources": {"cpus": 6}}}}})
    assert world.roles["main"].process.resources.cpus == 6


def test_world_override_can_clear_default_accelerators():
    @dryml.world.default(cpus=2, accelerators={"gpu": 1})
    def train():
        pass

    world = ann.resolve_world_default(train, overrides={"roles": {"main": {"process": {"resources": {"accelerators": {}}}}}})
    assert world.roles["main"].process.resources.accelerators == {}


def test_world_requirement_default_conflict_without_override_does_not_blame_override():
    @dryml.world.req(accelerators={"gpu": {"min": 1}})
    @dryml.world.default(cpus=2)
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    labels = {source.label for issue in result.report.issues for source in issue.sources}
    assert "user override" not in labels


def test_world_override_source_is_included_for_override_caused_invalid_spec():
    @dryml.world.default(cpus=2)
    def train():
        pass

    result = ann.resolve(train, overrides={"world": {"roles": {}}})
    assert not result.report.ok
    labels = {source.label for issue in result.report.issues for source in issue.sources}
    assert "user override" in labels


def test_world_zero_replicas_validates_instead_of_defaulting_to_one():
    @dryml.world.req(replicas=0)
    def train():
        pass

    req = ann.resolve_world_requirement(train)
    assert req.roles["main"].replicas.to_data() == {"exact": 0}

import dryml.annotations as ann
import dryml.env
import dryml.world


def test_conflicting_world_requirements_and_report_format():
    @dryml.world.req(replicas={"exact": 1})
    @dryml.world.req(replicas={"exact": 2})
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert result.report.issues[0].sources
    assert "[error]" in ann.format_report(result.report)


def test_conflicting_environment_fragments_include_sources():
    @dryml.env.req(python=">=3.10")
    @dryml.env.req(python=">=3.11")
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert result.report.issues[0].namespace == "environment"
    assert result.report.issues[0].sources

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


def test_conflicting_package_specifiers_report_issue():
    @dryml.env.req(requirements=("numpy<1",))
    @dryml.env.req(requirements=("numpy>=2",))
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert any(issue.path == "/requirements/numpy" for issue in result.report.issues)


def test_narrow_satisfiable_package_specifiers_do_not_report_false_conflict():
    @dryml.env.req(requirements=("numpy>1.0",))
    @dryml.env.req(requirements=("numpy<1.1",))
    def train():
        pass

    result = ann.resolve(train)
    assert result.report.ok


def test_invalid_environment_fragment_returns_structured_issue():
    @ann.require(namespace="environment", fragment={"requirements": ["not valid !!!"]})
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert result.report.issues[0].namespace == "environment"
    assert result.report.issues[0].sources


def test_error_on_conflict_merge_policy_reports_issue():
    @ann.default(namespace="runtime", fragment={"frameworks": {"torch": {"num_threads": 8}}}, merge_policy="error_on_conflict", priority=1)
    @ann.default(namespace="runtime", fragment={"frameworks": {"torch": {"num_threads": 4}}}, priority=0)
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert any(issue.path == "/frameworks/torch/num_threads" for issue in result.report.issues)


def test_environment_rejects_mapping_merge_policy():
    @ann.require(namespace="environment", fragment={"requirements": ["dryml"]}, merge_policy="replace")
    def train():
        pass

    result = ann.resolve(train)
    assert not result.report.ok
    assert any(issue.path == "/merge_policy" for issue in result.report.issues)

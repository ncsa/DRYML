"""Tests for diagnostic-first world requirement combination."""

import pytest

from dryml.annotations import Annotation, attach_annotation
from dryml.requirements import RequirementDeclaration, RequirementSource, require_admission
import dryml.worlds as worlds


def test_combines_inherited_and_selected_method_declarations_deterministically() -> None:
    """Reversed-C3 collection combines inherited world constraints once."""

    @worlds.req(cpus={"min": 1, "max": None}, source="base")
    class Base:
        pass

    @worlds.req(memory={"min": "1GiB", "max": None}, source="left")
    class Left(Base):
        pass

    @worlds.req(accelerators={"gpu": 1}, source="right")
    class Right(Base):
        pass

    class Target(Left, Right):
        @worlds.req(named={"license": 1}, source="method")
        def work(self):
            return None

    direct = worlds.requirements_for(Target)
    selected = worlds.requirements_for_method(Target(), "work")
    role = direct.value.roles["main"]
    assert role.resources.cpus == worlds.CountConstraint(1, None)
    assert role.resources.memory == worlds.CountConstraint(1024**3, None)
    assert role.resources.accelerators["gpu"] == worlds.CountConstraint(1, 1)
    assert selected.value.roles["main"].resources.named["license"] == worlds.CountConstraint(1, 1)
    assert worlds.requirements_for_method(Target, "work") == selected


def test_reports_all_resource_and_topology_conflicts_with_sources() -> None:
    """Independent role paths report complete conflicts with every contributor."""

    @worlds.req(replicas=1, cpus=1, memory="1GiB", accelerators={"gpu": 1}, accelerator_memory={"gpu": "1GiB"}, devices={"fpga": 1}, named={"license": 1}, topology={"rack": "a"}, source="first")
    class First:
        pass

    @worlds.req(replicas=2, cpus=2, memory="2GiB", accelerators={"gpu": 2}, accelerator_memory={"gpu": "2GiB"}, devices={"fpga": 2}, named={"license": 2}, topology={"rack": "b"}, source="second")
    class Second(First):
        pass

    result = worlds.requirements_for(Second)
    assert result.value is None
    assert [issue.path for issue in result.report.issues] == [
        "roles.main.replicas",
        "roles.main.resources.accelerator_memory.gpu",
        "roles.main.resources.accelerators.gpu",
        "roles.main.resources.cpus",
        "roles.main.resources.devices.fpga",
        "roles.main.resources.memory",
        "roles.main.resources.named.license",
        "roles.main.topology.rack",
    ]
    assert all(tuple(source.label for source in issue.sources) == ("1: first", "2: second") for issue in result.report.issues)


def test_dotted_role_and_resource_names_remain_distinct_and_attributed() -> None:
    """Structured conflict paths distinguish dotted names at different boundaries."""

    class Subject:
        pass

    declarations = (
        ("dotted-role-first", "main.resources.named", "license", 1),
        ("dotted-role-second", "main.resources.named", "license", 2),
        ("dotted-resource-first", "main", "named.resources.named.license", 1),
        ("dotted-resource-second", "main", "named.resources.named.license", 2),
    )
    for label, role, resource, count in declarations:
        attach_annotation(
            Subject,
            Annotation(
                worlds.WORLD_REQUIREMENT_KEY,
                RequirementDeclaration(
                    worlds.WorldRequirement({role: {"resources": {"named": {resource: count}}}}),
                    source=RequirementSource(label),
                ),
            ),
        )

    result = worlds.requirements_for(Subject)
    repeated = worlds.requirements_for(Subject)
    issues = {issue.path: issue for issue in result.report.issues}

    assert result == repeated
    assert set(issues) == {
        'roles.main.resources.named["named.resources.named.license"]',
        'roles["main.resources.named"].resources.named.license',
    }
    assert tuple(source.label for source in issues['roles.main.resources.named["named.resources.named.license"]'].sources) == (
        "3: dotted-resource-first",
        "4: dotted-resource-second",
    )
    assert tuple(source.label for source in issues['roles["main.resources.named"].resources.named.license'].sources) == (
        "1: dotted-role-first",
        "2: dotted-role-second",
    )
    assert all(len(path) <= 512 for path in issues)


def test_malformed_attached_values_and_over_budget_combinations_fail_before_result() -> None:
    """Corrupt metadata and preflight capacity failures cannot expose a partial value."""

    def target():
        return None

    attach_annotation(target, Annotation(worlds.WORLD_REQUIREMENT_KEY, "bad"))
    with pytest.raises(worlds.WorldRequirementError):
        worlds.requirements_for(target)

    class Subject:
        pass

    for index in range(257):
        attach_annotation(Subject, Annotation(worlds.WORLD_REQUIREMENT_KEY, RequirementDeclaration(worlds.WorldRequirement({f"role_{index}": {}}), source=RequirementSource(str(index)))))
    with pytest.raises(worlds.WorldRequirementError):
        worlds.requirements_for(Subject)


def test_public_merge_methods_share_world_semantics() -> None:
    """Legacy two-value merge APIs retain compatible and conflicting behavior."""

    merged = worlds.WorldRequirement({"main": {"resources": {"cpus": {"min": 1, "max": None}}}}).merge(
        worlds.WorldRequirement({"main": {"resources": {"cpus": {"min": None, "max": 2}}}})
    )
    assert merged.roles["main"].resources.cpus == worlds.CountConstraint(1, 2)
    with pytest.raises(worlds.ResourceValidationError):
        worlds.CountConstraint(1, 1).merge(worlds.CountConstraint(2, 2))
    with pytest.raises(worlds.ResourceValidationError):
        worlds.WorldRequirement({"main": {"resources": {"cpus": 1}}}).merge(
            worlds.WorldRequirement({"main": {"resources": {"cpus": 2}}})
        )


def test_compatibility_reports_admit_with_their_existing_ok_decision() -> None:
    """World compatibility retains the policy-independent hard admission signal."""

    requirement = worlds.WorldRequirement({"main": {"resources": {"cpus": 1}}})
    compatible = worlds.check_world_spec_satisfies_requirement(worlds.WorldSpec.from_payload({"roles": {"main": {"process": {"resources": {"cpus": 1}}}}}), requirement)
    incompatible = worlds.check_world_spec_satisfies_requirement(worlds.WorldSpec.from_payload({"roles": {"main": {"process": {"resources": {"cpus": 0}}}}}), requirement)
    assert compatible.admission_ok is compatible.ok is True
    assert incompatible.admission_ok is incompatible.ok is False
    assert require_admission(compatible) is None
    with pytest.raises(Exception) as raised:
        require_admission(incompatible)
    assert raised.value.report is incompatible


def test_capacity_and_redaction_preflight_is_complete_or_fails_before_result() -> None:
    """Accepted diagnostic work is complete while oversized work has no result."""

    def requirement(count: int, *, offset: int = 0) -> worlds.WorldRequirement:
        role = worlds.RoleRequirement(
            replicas=worlds.CountConstraint(),
            resources=worlds.ResourceRequirement.from_data({"cpus": 1}),
        )
        return worlds.WorldRequirement(
            {
                f"role_{index + offset}": role
                for index in range(count)
            }
        )

    class Boundary:
        pass

    for label in range(4):
        attach_annotation(
            Boundary,
            Annotation(
                worlds.WORLD_REQUIREMENT_KEY,
                RequirementDeclaration(requirement(1024), source=RequirementSource(str(label))),
            ),
        )
    assert worlds.requirements_for(Boundary).has_value

    class OverBudget:
        pass

    for label in range(5):
        attach_annotation(
            OverBudget,
            Annotation(
                worlds.WORLD_REQUIREMENT_KEY,
                RequirementDeclaration(requirement(1024), source=RequirementSource(str(label))),
            ),
        )
    with pytest.raises(worlds.WorldRequirementError):
        worlds.requirements_for(OverBudget)

    @worlds.req(named={"token": 1}, topology={"/private/path": "a"}, source="first")
    class First:
        pass

    @worlds.req(named={"token": 2}, topology={"/private/path": "b"}, source="second")
    class Second(First):
        pass

    paths = [issue.path for issue in worlds.requirements_for(Second).report.issues]
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert all("token" not in path and "/private/path" not in path for path in paths)


def test_single_full_envelope_world_value_passes_through_unchanged() -> None:
    """One existing full world value bypasses aggregate diagnostic capacities."""

    value = worlds.WorldRequirement({f"role_{index}": {} for index in range(4096)})

    @worlds.req(roles=value.roles)
    class Target:
        pass

    assert worlds.WorldRequirement.from_data(value.to_data()) == value
    result = worlds.requirements_for(Target)
    assert result.value == value
    assert result.value.semantic_id == value.semantic_id

import dryml.environments as envs


def test_registry_register_get_find_and_serialize():
    registry = envs.EnvironmentRegistry()
    spec = envs.CurrentEnvironmentSpec()
    entry = registry.register("current", spec, provides=("dryml.environments.v1",), tags=("dev",))
    assert registry.get("current") == entry
    assert registry.list() == (entry,)
    req = envs.EnvironmentRequirement(capabilities=("dryml.environments.v1",), tags=("dev",))
    assert registry.find(req) == entry
    clone = envs.EnvironmentRegistry.from_data(registry.to_data())
    assert clone.get("current").to_data() == entry.to_data()


def test_registry_duplicate_and_missing_name_errors():
    registry = envs.EnvironmentRegistry()
    registry.register("current", envs.CurrentEnvironmentSpec())
    try:
        registry.register("current", envs.CurrentEnvironmentSpec())
    except envs.EnvironmentRegistryError as exc:
        assert exc.context["name"] == "current"
    else:
        raise AssertionError("expected duplicate registry error")
    try:
        registry.get("missing")
    except envs.EnvironmentRegistryError as exc:
        assert exc.context["name"] == "missing"
    else:
        raise AssertionError("expected missing registry error")


def test_registry_unregister_is_deterministic_and_probe_free():
    registry = envs.EnvironmentRegistry()
    entry = registry.register("worker", envs.CurrentEnvironmentSpec())
    assert registry.unregister("worker") == entry
    try:
        registry.unregister("worker")
    except envs.EnvironmentRegistryError as exc:
        assert exc.context["name"] == "worker"
    else:
        raise AssertionError("expected missing registry error")


def test_registry_probe_and_find_compatible():
    registry = envs.EnvironmentRegistry()
    registry.register("current", envs.CurrentEnvironmentSpec(), tags=("current",))
    result = registry.probe_registered("current")
    assert result.ok
    entry, report = registry.find_compatible(envs.EnvironmentRequirement(tags=("current",)), timeout=30)
    assert entry.name == "current"
    assert report.ok
    missing, missing_report = registry.find_compatible(envs.EnvironmentRequirement(tags=("missing",)))
    assert missing is None
    assert not missing_report.ok


def test_registry_check_requirement_and_no_match_report():
    registry = envs.EnvironmentRegistry()
    registry.register("current", envs.CurrentEnvironmentSpec())
    report = registry.check_requirement("current", envs.EnvironmentRequirement(capabilities=("dryml.environments.v1",)))
    assert report.ok
    no_match = registry.no_match_report(envs.EnvironmentRequirement(requirements=("torch",)))
    assert no_match.issues[0].code == "registry_no_match"

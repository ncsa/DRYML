"""Closed merge policy, priority, conflict, and clear coverage."""

from dryml.annotations import AnnotationFragment, AnnotationTarget, SourceTrace, resolve_fragments


def _fragment(value, *, policy=None, priority=0, kind="default", source="source"):
    return AnnotationFragment(AnnotationTarget("synthetic", None, source), "runtime", kind, value, SourceTrace("synthetic", label=source, namespace="runtime"), priority, policy)


def test_runtime_mapping_merges_recursively_and_conflicting_hard_requirements_name_both_sources():
    result = resolve_fragments((_fragment({"limits": {"threads": 1}}, kind="requirement", source="left"), _fragment({"limits": {"threads": 2}}, kind="requirement", source="right")))
    assert not result.usable
    assert [source.label for source in result.diagnostics[0].sources] == ["left", "right"]


def test_replace_append_error_conflict_clear_and_priority_rules():
    replace = resolve_fragments((_fragment({"a": 1}), _fragment({"a": 2}, policy="replace")))
    assert replace.runtime_default["a"] == 2
    appended = resolve_fragments((_fragment({"items": [1]}), _fragment({"items": [2]}, policy="append")))
    assert appended.runtime_default["items"] == (1, 2)
    rejected = resolve_fragments((_fragment({"a": 1}), _fragment({"a": 2}, policy="error_on_conflict")))
    assert not rejected.usable
    rebuilt = resolve_fragments((_fragment({"a": 1}), _fragment({}, policy="clear"), _fragment({"b": 2})))
    assert rebuilt.runtime_default == {"b": 2}
    priority = resolve_fragments((_fragment({"a": 1}, policy="replace", priority=10), _fragment({"a": 2}, policy="replace", priority=0)))
    assert priority.runtime_default["a"] == 1


def test_initial_append_rejects_non_sequence_leaves_and_display_redacts_sources():
    """Append validates its first value and display data hides source paths."""

    invalid = resolve_fragments((_fragment({"value": 1}, policy="append"),))
    assert not invalid.usable
    source = SourceTrace("synthetic", label="source", namespace="runtime", path="/private/source.py", metadata={"api_token": "secret"})
    fragment = AnnotationFragment(AnnotationTarget("synthetic", None, "source"), "runtime", "default", {"items": [1]}, source, 0, "append")
    result = resolve_fragments((fragment,))
    assert result.usable
    assert result.to_data()["source_traces"][0]["path"] == "<local-path>"
    assert result.to_data()["source_traces"][0]["metadata"]["api_token"] == "<redacted>"

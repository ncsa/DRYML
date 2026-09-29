"""Closed quotation and admission coverage for TemplateBundle."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from dryml.core import Definition, Object, Par, Ref, Repo, Template, TemplateBundle
from dryml.core.cdef_codec import decode_cdef_graph, encode_cdef_graph
from dryml.core.errors import TemplateError, TemplateLimitError
from dryml.core.store.dir import DirStore
from tests.qualification_fixture_support import verify_qualification_fixture_manifest


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "qualification_reader_v1"


class BundleLeaf(Object):
    """Inert portable target proving bundle handling does not construct recipes."""

    constructed = 0

    def __init__(self, value):
        type(self).constructed += 1
        self.value = value


class BundleOwner(Object):
    """Owner explicitly admitting the aggregate recipe quotation carrier."""

    def __init__(self, artifacts: Ref[TemplateBundle | None] = None):
        self.artifacts = artifacts


class RawBundleOwner(Object):
    """Owner used to prove an ordinary field cannot carry a bundle."""

    def __init__(self, artifacts):
        self.artifacts = artifacts


def _recipe(value: int) -> Template:
    """Return one unresolved-free portable recipe without constructing its target."""

    return Template(BundleLeaf, value)


def test_bundle_normalizes_public_forms_without_recipe_execution():
    """Singleton, sequence, and mapping forms retain frozen deterministic names."""

    BundleLeaf.constructed = 0
    singleton = TemplateBundle(_recipe(1))
    sequence = TemplateBundle([_recipe(1), _recipe(2)])
    mapping = TemplateBundle({"metric.accuracy": _recipe(1), "loss/value": _recipe(2)})

    assert singleton.names == ("artifact_0",)
    assert sequence.names == ("artifact_0", "artifact_1")
    assert mapping.names == ("metric.accuracy", "loss/value")
    assert BundleLeaf.constructed == 0


def test_bundle_ref_admission_codec_and_store_restore_remain_inert(tmp_path):
    """Only Ref admission persists one canonical aggregate recipe quotation."""

    BundleLeaf.constructed = 0
    bundle = TemplateBundle({"metric.accuracy": _recipe(1), "metric.loss": _recipe(2)})
    owner = Definition(BundleOwner, bundle).concretize()
    restored_cdef = decode_cdef_graph(encode_cdef_graph(owner))
    restored_bundle = restored_cdef.parameters["artifacts"].target
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save_object(BundleOwner(bundle, repo=repo))
    restored = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert restored_bundle.to_data() == bundle.to_data()
    assert restored.artifacts.names == bundle.names
    assert BundleLeaf.constructed == 0
    with pytest.raises((TypeError, TemplateError)):
        Definition(RawBundleOwner, bundle).concretize()


def test_bundle_payload_rejects_wrong_kind_duplicate_name_and_noncanonical_data():
    """Closed aggregate payloads cannot bypass role or canonical ordering checks."""

    bundle = TemplateBundle({"first": _recipe(1), "second": _recipe(2)})
    payload = bundle.to_data()
    wrong_kind = deepcopy(payload)
    wrong_kind["kind"] = "template"
    noncanonical = deepcopy(payload)
    noncanonical["nodes"].reverse()

    with pytest.raises(TemplateError):
        TemplateBundle.from_data(wrong_kind)
    with pytest.raises(TemplateError, match="canonical"):
        TemplateBundle.from_data(noncanonical)
    with pytest.raises((TypeError, TemplateError)):
        TemplateBundle({"": _recipe(1)})
    with pytest.raises(TypeError):
        TemplateBundle([_recipe(1), object()])


def test_bundle_v1_fixture_round_trips_without_resolving_its_recipe():
    """Keep the committed aggregate recipe fixture independent of Artifact payloads."""

    verify_qualification_fixture_manifest(FIXTURE_ROOT)
    payload = json.loads((FIXTURE_ROOT / "template_bundle.json").read_text(encoding="ascii"))
    restored = TemplateBundle.from_data(payload)

    assert restored.to_data() == payload
    assert restored.names == ("score",)


def test_bundle_ref_is_opaque_by_default_and_opt_in_traversal_enters_each_recipe():
    """Binding and remapping cross a bundle Ref only when explicitly requested."""

    bundle = TemplateBundle({
        "first": Template(BundleLeaf, Par("width")),
        "second": Template(BundleLeaf, Par("height")),
    })
    outer = Template.from_value({"artifacts": Ref(bundle)})

    with pytest.raises(TemplateError, match="unknown"):
        outer.sub(width=4)

    bound = outer.sub(sub_dict={"width": 4, "height": 8}, traverse_refs=True)
    remapped = outer.remap(prefix="metric", traverse_refs=True)
    bound_bundle = bound.root["artifacts"].target
    remapped_bundle = remapped.root["artifacts"].target

    assert bound_bundle.recipes["first"].root.args[0] == 4
    assert bound_bundle.recipes["second"].root.args[0] == 8
    assert remapped_bundle.recipes["first"].root.args[0] == Par("metric/width")
    assert remapped_bundle.recipes["second"].root.args[0] == Par("metric/height")


def test_bundle_codec_applies_one_aggregate_entry_budget(monkeypatch):
    """Individually valid recipes cannot reset the aggregate payload budget."""

    from dryml.core import template_codec

    bundle = TemplateBundle({
        "first": Template.from_value([0] * 2_100),
        "second": Template.from_value([1] * 2_100),
    })
    with pytest.raises(TemplateLimitError, match="aggregate entry"):
        bundle.to_data()

    monkeypatch.setattr(template_codec, "_MAX_ENTRIES", 5_000)
    payload = bundle.to_data()
    monkeypatch.setattr(template_codec, "_MAX_ENTRIES", 4_096)
    with pytest.raises(TemplateLimitError, match="aggregate entry"):
        TemplateBundle.from_data(payload)


def test_bundle_codec_counts_inline_source_import_entries(monkeypatch):
    """Inline source dependencies cannot bypass the aggregate bundle budget."""

    from dryml.core import ImportRef, SourceSpec, template_codec

    reference = ImportRef("builtins", "int")
    imports = lambda prefix: {
        f"{prefix}_{index}": reference for index in range(2_100)
    }
    bundle = TemplateBundle({
        "first": Template.from_value(SourceSpec("function", "lambda: None", imports=imports("a"))),
        "second": Template.from_value(SourceSpec("function", "lambda: None", imports=imports("b"))),
    })
    with pytest.raises(TemplateLimitError, match="aggregate entry"):
        bundle.to_data()

    monkeypatch.setattr(template_codec, "_MAX_ENTRIES", 5_000)
    payload = bundle.to_data()
    monkeypatch.setattr(template_codec, "_MAX_ENTRIES", 4_096)
    with pytest.raises(TemplateLimitError, match="aggregate entry"):
        TemplateBundle.from_data(payload)

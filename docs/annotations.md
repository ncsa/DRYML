# Annotations

`dryml.annotations` is a passive, process-local metadata kernel. It attaches
consumer-owned declarations to live Python targets and collects them in a
deterministic order. It does not resolve requirements, select Methods, enforce
runtime policy, wrap calls, launch work, or activate a framework.

## Carrier And Keys

An `Annotation` is a frozen two-field carrier: a consumer-selected `key` and an
opaque consumer-owned `value`. Its equality and hash are identity-based. The
kernel retains the exact value object without copying, deep-freezing, comparing,
hashing, serializing, or assigning it an ID. The frozen carrier provides only
shallow immutability: consumers must supply and treat their values as immutable.

Keys are exact built-in ASCII strings of 1 through 128 characters matching
`[A-Za-z_][A-Za-z0-9_.-]*`. They are compared exactly; the kernel does not keep
a registry or assign meaning to them. Use owner-qualified keys such as
`"example.policy"` to avoid accidental matches with another consumer.

```python
from dryml.annotations import Annotation, attach_annotation, collect_annotations


def policy(name):
    entry = Annotation("example.policy", name)

    def decorate(target):
        return attach_annotation(target, entry)

    return decorate


@policy("fast")
def train():
    return "trained"


assert train() == "trained"
assert collect_annotations(train, key="example.policy")[0].value == "fast"
```

The example decorator is consumer-owned. `dryml.annotations` supplies no
domain-specific decorators, including Method, requirement, environment, world,
or runtime decorators. Adoption by those owners remains separate work.

## Attachment Lifecycle

`attach_annotation(target, annotation)` appends one entry to the direct
`__dryml_annotations__` tuple and returns the exact same target. It never wraps,
binds, calls, replaces, or imports through the target.

Supported targets are extensible Python functions, classes, `staticmethod` and
`classmethod` descriptors, and custom descriptors with a real instance
dictionary and native `object.__setattr__`. Properties, builtins, and
descriptors that override attribute mutation are rejected without mutation. If
the final native assignment fails for an otherwise statically safe target, such
as an immutable built-in class, attachment raises
`UnsupportedAnnotationTargetError` with the native failure as its cause. A
target that defines a data descriptor at the reserved
`__dryml_annotations__` attribute is also rejected so attachment cannot invoke
target-owned getter or setter behavior. `own_annotations(target)` returns only
the exact target's direct built-in tuple containing exact `Annotation` entries
in declaration order; it does not inherit or unwrap entries.

Attach declarations during import, class definition, or setup before sharing a
target. Concurrent writes to the same target are unsupported. Concurrent
read-only collection after setup is supported for an unchanged target. Entries
are process-local: a spawned worker or fresh interpreter must independently
import or reconstruct and annotate its targets. The kernel provides no
cross-process annotation transport.

`AnnotationValidationError` reports invalid keys, invalid carriers, malformed
direct metadata, and malformed collector input. `UnsupportedAnnotationTargetError`
reports targets that cannot be safely inspected or mutated. Both derive from
`AnnotationError`; failed validation or target eligibility leaves existing
metadata unchanged.

## Collection

`collect_annotations(target)` returns an immutable tuple. For a class it
collects direct declarations in reversed C3 order, from base to subclass,
excluding `object`; each class retains its direct declaration order. For a
supported non-class target it returns direct entries. A `staticmethod` or
`classmethod` descriptor contributes its direct entries followed by those of
its known underlying function.

`annotations_for_method(cls, name)` first returns the class sequence, then the
descriptor selected by normal static MRO lookup, then its known underlying
function. An overridden base implementation that is not selected is excluded.
`annotations_for_members(cls, key=None, after=None)` instead returns immutable
`AnnotatedMember(owner, name, descriptor, annotations)` evidence for every
matching member declaration in base-to-subclass C3 order. `owner` is the exact
declaring class, `name` is the declared member name, and `descriptor` is the raw
unbound namespace value. Its annotation tuple contains direct descriptor entries
and, for known `staticmethod` and `classmethod` descriptors only, direct entries
on the underlying function. A safe custom descriptor contributes only its direct
entries. A later unannotated declaration with a name that previously matched is
also returned with an empty annotation tuple so consumers can interpret an
override or shadow themselves.

When `after` names a class in `cls`'s MRO, that boundary and all of its base
classes are excluded. Collection starts at the next subclass, retaining the same
ordering and shadow semantics. An invalid or unrelated boundary raises
`AnnotationValidationError` before inspection.

Collection never binds descriptors or invokes `__get__`, dynamic attribute
hooks, properties, imports, or user code. If the same `Annotation` object is
seen through multiple paths, it appears once at its first occurrence; distinct
objects remain distinct even when their keys and values match. The same
identity-deduplication rule applies within each `AnnotatedMember` annotation
tuple. The member collector returns no partial result: unsupported descriptor
members and malformed direct metadata raise the existing generic annotation
errors.

All collectors accept `key=` for exact-key filtering. Filtering preserves the
unfiltered relative order and identity deduplication. For member collection,
matching and shadow evidence are evaluated after filtering. Invalid filter keys,
missing members, unsupported targets, and malformed direct metadata raise the
generic annotation errors instead of returning partial results.

## Ownership And Clean Break

The kernel does not serialize or persist attached entries, create annotation
envelopes or content IDs, collect Definition-specific metadata, retain source
diagnostics, or interpret values. After collection, the consumer owns semantic
combination and may produce a separate consumer-owned derived value with its
own persistence policy.

This clean break removes the old fragment, requirement/default, merge,
resolution, source-diagnostic, envelope/ID, namespace, and Definition helper
APIs from `dryml.annotations`, including `AnnotationFragment`, `FRAGMENT_ATTR`,
`attach_fragment`, `collect_fragments`, `own_fragments`, `require`, `default`,
`resolve_fragments`, and `fragments_for_definition_method`. The retired
`dryml.annotations.decorators`, `env`, `world`, `runtime`, `merge`,
`namespaces`, and `storage` modules are not importable. Root `dryml.env` and
`dryml.world`, plus `dryml.runtime.default`, are also removed.

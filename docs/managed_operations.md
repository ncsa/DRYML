# Managed Operations

`dryml.managed` begins with checked declaration and input-matching support for
synchronous instance methods. Use `@managed_operation(resumable=...)` with a
required keyword-only `managed` parameter, and pass caller policy through
`ManagedConfig`.

The internal U4 persistence layer resolves explicit `DirStore` bindings, or an
omitted state Store from `dryml.core.session.current_repo`: one physical Store is
selected directly, while multiple physical Stores require exactly one full match
for the live object's exact current `StateRef`. Omitted control authority uses
the selected state Store. Resolution validates publication capabilities before
mutation and never copies state, migrates control, or records a control locator.

Managed control authority is a closed, bounded JSON layout rooted at
`managed/` in the selected control Store. It uses a versioned format gate,
generation-numbered current snapshots, a stable control lock, staged initial
operation directories, and a transient pending replacement intent. Inspection
does not initialize an absent namespace. A malformed gate, missing current file
inside an existing operation directory, incomplete referenced StateRef, or
pending replacement is a recovery/control error rather than a fabricated status.
Pending intents and replacement snapshots are first flushed as same-directory
temporary files, then atomically published. POSIX publication also synchronizes
the parent directory; on Windows the adapter synchronizes regular files and uses
atomic replacement, without claiming unsupported directory-descriptor fsync.

This unit does not yet execute methods, acquire graph-lifetime ownership,
checkpoint state, expose status, or request interruption. Those lifecycle
behaviors remain deferred to U5-U7; the U4 primitives are intentionally narrow
so those units can compose them without introducing histories, journals,
locators, or a generic records layer.

# Managed Operations

`dryml.managed` begins with checked declaration and input-matching support for
synchronous instance methods. Use `@managed_operation(resumable=...)` with a
required keyword-only `managed` parameter, and pass caller policy through
`ManagedConfig`.

This U3 surface validates declarations, store/config inputs, and stable operation
and argument identities. It does not yet execute methods, create lifecycle
authority, checkpoint state, inspect status, or request interruption; those
lifecycle behaviors are deferred to subsequent Stage 6 units.

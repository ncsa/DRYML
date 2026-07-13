# DRYML Repo Policies

## Documentation
Primary documentation is via docstrings, keep those updated for all classes/methods. Docstrings should detail what classes are responsible for and method docstrings should detail their actions as well as arguments/types and return types.

There is a `docs` directory containing explanatory material showing how to use the API. Keep this up to date as well. That may include new .md files, new sections, or editing existing content.

## Tests
Running the test suite is handled with `tests.sh`. Ignore tests in `tests/{old,dev}` they are currently old. I usually execute like this: `./tests.sh --ignore tests/old --ignore tests/dev -x tests`. The script passes additional arguments through, and you can run focused tests by specifying the files or folders you want to run.

Try to avoid executing the full test suite often. There are some tests which are very 'heavy' and take a long time to run. `tests/core` is okay to run.

## Directory explanation

### `src/dryml` - The main source code repository
`core2` - core modules of dryml.

`code` - utilities for method instrumentation
`execute` - The remote execution subsystem of DRYML
`context` - The compute context subsystem of DRYML
`graph` - Generic graph algorithms. Used in various places by DRYML
`data` - The Dataset API submodule
`models` - The Model API submodule
`artifacts` - The Artifacts API submodule
`vis` - A collection of useful visualization methods that integrate with DRYML
`devtools` - Tools to help the user while developing with DRYML for example in a jupyter notebook.

### Plugins
`ray` - ray specific DRYML plugin
`jax` - jax specific DRYML plugin
`tf` - tf specific DRYML plugin
`torch` - torch specific DRYML plugin

## Commit Workflow

After every verified coherent set of implementation, test, documentation, or policy changes, commit it unless the user explicitly says not to commit, verification fails, or the work remains incomplete:

1. Inspect `git status`, the relevant `git diff`, and `git log --oneline -10` before staging.
2. Stage only the intended files for that commit.
3. Review the staged changes with `git diff --cached` and verify they contain no unrelated or private data.
4. Draft a concise repository-style commit message that describes the changes and relevant verification in a temporary owner-only file outside the repository, such as `/tmp/opencode/<commit>-message.txt`.
5. Commit with `git commit -F <temporary-file>`.
6. Remove the temporary message file only after the commit succeeds; retain it when the commit fails.
7. Report the resulting commit SHA and any required parent-repository submodule-pointer follow-up.

Do not amend, push, broadly stage, reset, discard, stash, or rewrite history. Leave incomplete or failed work uncommitted and report the blocker.

### Other
`examples` - Example dryml programs to illustrate dryml use cases

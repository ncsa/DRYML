# DRYML Repo Policies

## Documentation
Primary documentation is via docstrings, keep those updated for all classes/methods. Docstrings should detail what classes are responsible for and method docstrings should detail their actions as well as arguments/types and return types.

There is a `docs` directory containing explanatory material showing how to use the API. Keep this up to date as well. That may include new .md files, new sections, or editing existing content.

## Tests
Running the test suite is handled with `tests.sh`. Ignore tests in `tests/{old,dev}` they are currently old. I usually execute like this: `./tests.sh --ignore tests/old --ignore tests/dev -x tests`. The script passes additional arguments through, and you can run focused tests by specifying the files or folders you want to run.

Try to avoid executing the full test suite often. There are some tests which are very 'heavy' and take a long time to run. `tests/core` is okay to run.

## Directory explanation

### `src/dryml` - The main source code repository
`core` - core modules of dryml.

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

## Draft Commit Message

A draft commit message should be written to `COMMIT-MSG`. This file may already exist in which case it includes a draft already in progress. Append your new version after the current text. The user will make final edits given the history of the evolving message. Check the last commit message before editing. A previous draft message may have already been checked in. In which case, inspect the diff and make the commit message reflect that diff.

### Other
`examples` - Example dryml programs to illustrate dryml use cases

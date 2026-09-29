# Release Process

DRYML publishes one pure-Python wheel. Release automation uses GitHub trusted
publishing, so no PyPI API token is stored in GitHub.

## One-Time Configuration

Create GitHub environments named `testpypi` and `pypi`. Protect `pypi` with the
required reviewer and deployment-branch or tag rules appropriate for the
repository. Protect the `test-v*` and `v*` tag namespaces against unauthorized
creation or replacement.

Configure trusted publishers for the `ncsa/dryml` project on both package
indexes:

| Index | Workflow | GitHub environment |
| --- | --- | --- |
| TestPyPI | `release-rehearsal.yaml` | `testpypi` |
| PyPI | `release.yaml` | `pypi` |

The publisher owner is `ncsa` and the repository is `dryml`. The workflows need
only GitHub's short-lived OIDC identity; do not add PyPI passwords or tokens.

## Rehearsal

Set the intended version in `setup.cfg`, merge the release commit, and wait for
its normal CI checks to pass. The rehearsal tag is exactly
`test-v<package-version>`. For version `0.3.0b1`:

```bash
git tag -a test-v0.3.0b1 -m "Rehearse 0.3.0b1"
git push origin test-v0.3.0b1
```

`.github/workflows/release-rehearsal.yaml` checks that the tag suffix exactly
matches the package version, builds and validates one wheel, retains it as a
workflow artifact, and publishes it to TestPyPI. Index versions are immutable;
a repeated rehearsal requires a new package version and tag rather than
overwriting an upload.

Inspect the TestPyPI project page and test installation against TestPyPI before
creating the production tag. TestPyPI does not mirror every dependency, so use
PyPI as the dependency fallback:

```bash
python -m venv /tmp/dryml/release-test
/tmp/dryml/release-test/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  dryml==0.3.0b1
```

## Official Release

After the rehearsal succeeds, tag the same reviewed commit with exactly
`v<package-version>` and push it:

```bash
git tag -a v0.3.0b1 -m "Release 0.3.0b1"
git push origin v0.3.0b1
```

`.github/workflows/release.yaml` validates the tag, builds and checks one wheel,
and publishes that wheel to PyPI through the protected `pypi` environment. Only
after PyPI accepts the wheel does the workflow create the GitHub Release, using
generated release notes and attaching the exact wheel passed between jobs as a
GitHub Actions artifact.

If PyPI publication succeeds but GitHub Release creation fails, do not replace
or move the tag and do not republish the immutable PyPI file. Download the
`release-wheel` workflow artifact and create the GitHub Release for the existing
tag with that exact wheel.

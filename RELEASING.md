# Releasing PyJAGS

## Automated Release (recommended)

Pushing a version tag triggers GitHub Actions to build wheels for all platforms
and publish to PyPI automatically via trusted publishing.

### Steps

1. **Ensure tests pass** — check that CI is green on master.

2. **Tag the release** — tags have no prefix (e.g. `2.1.0`, not `v2.1.0`):

   ```bash
   git tag X.Y.Z
   ```

3. **Push the tag**:

   ```bash
   git push origin master
   git push origin X.Y.Z
   ```

4. **Monitor the release workflow** at
   https://github.com/michaelnowotny/pyjags/actions/workflows/release.yml

   The workflow builds:
   - Source distribution (sdist)
   - Linux wheels (x86_64, aarch64) for Python 3.12 and 3.13
   - macOS wheels (arm64, x86_64) for Python 3.12 and 3.13

   On success, all artifacts are uploaded to PyPI automatically.

5. **Create a GitHub Release** at
   https://github.com/michaelnowotny/pyjags/releases/new — select the existing
   tag and write release notes.

### One-time setup: PyPI trusted publishing

Before the first automated release, configure trusted publishing on PyPI:

1. Go to https://pypi.org/manage/project/pyjags/settings/publishing/
2. Add a new publisher:
   - Owner: `michaelnowotny`
   - Repository: `pyjags`
   - Workflow: `release.yml`
   - Environment: `pypi`

### Dry run (no publish)

Use the "Run workflow" button on the Actions page to trigger a build without
publishing. This runs on `workflow_dispatch` and skips the publish step.

## Manual Release

If you need to release without GitHub Actions (e.g. for a hotfix):

```bash
pip install twine build

# Build
rm -rf dist/ build/
python -m build --sdist

# Upload to Test PyPI first
twine upload --repository-url https://test.pypi.org/legacy/ dist/pyjags-X.Y.Z.tar.gz

# Verify, then upload to real PyPI
twine upload dist/pyjags-X.Y.Z.tar.gz
```

For manual uploads, you need a PyPI API token. Create one at
https://pypi.org/manage/account/token/ (scoped to the `pyjags` project).

## Rollback

If a critical issue is discovered after upload:

1. **Yank** the release on PyPI (hides from `pip install pyjags` but allows
   `pip install pyjags==X.Y.Z`).
2. Fix the issue, tag `X.Y.Z+1`, rebuild, and upload.
3. **Do not delete** the PyPI release — that version number is permanently
   burned.
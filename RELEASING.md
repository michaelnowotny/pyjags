# Releasing PyJAGS

Steps to publish a new release to PyPI.

## Prerequisites

```bash
pip install twine build
```

You will need API tokens for [PyPI](https://pypi.org/manage/account/token/) and
optionally [Test PyPI](https://test.pypi.org/manage/account/token/). Scope the
PyPI token to the `pyjags` project. Store tokens securely (e.g. in a password
manager or `~/.pypirc`).

## 1. Ensure tests pass

```bash
python -m pytest test/ -v

# Optional: multi-Python matrix via Docker
./scripts/test-all-pythons
```

## 2. Tag the release

Tags have no prefix (e.g. `2.1.0`, not `v2.1.0`). setuptools-scm infers the
version from git tags.

```bash
git tag X.Y.Z
```

Verify the version is picked up:

```bash
pip install -e . && python -c "import pyjags; print(pyjags.__version__)"
# Should print: X.Y.Z
```

## 3. Push the tag

```bash
git push origin master
git push origin X.Y.Z
```

## 4. Build the source distribution

```bash
rm -rf dist/ build/

python -m build --sdist

# Verify
ls -la dist/
# Should show: pyjags-X.Y.Z.tar.gz
```

## 5. Upload to Test PyPI (recommended)

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/pyjags-X.Y.Z.tar.gz
```

Verify:

```bash
python -m venv /tmp/test-pyjags
source /tmp/test-pyjags/bin/activate

pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pyjags==X.Y.Z

python -c "import pyjags; print(pyjags.__version__)"

deactivate
rm -rf /tmp/test-pyjags
```

## 6. Upload to PyPI

Only proceed if Test PyPI verification passed.

```bash
twine upload dist/pyjags-X.Y.Z.tar.gz
```

## 7. Create a GitHub Release

Go to https://github.com/michaelnowotny/pyjags/releases/new, select the
existing tag, and write release notes.

## Credentials

You can avoid entering credentials each time by creating `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-real-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-test-token>
```

Set `chmod 600 ~/.pypirc` to protect your tokens.

## Rollback

If a critical issue is discovered after upload:

1. **Yank** the release on PyPI (hides from `pip install pyjags` but allows
   `pip install pyjags==X.Y.Z`).
2. Fix the issue, tag `X.Y.Z+1`, rebuild, and upload.
3. **Do not delete** the PyPI release — that version number is permanently
   burned.
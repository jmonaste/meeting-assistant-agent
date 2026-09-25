# wheelhouse/

Optional. If this folder contains `.whl` files when the image is built, the
Dockerfile installs **only** from here (`pip install --no-index`), so the build
needs no access to PyPI or to a mirror. Leave it empty for a normal online build.

How to fill it from a machine that can already `pip install` the project (for
example the laptop where the CLI works) is described in
[docs/08-Web-UI-And-OpenShift.md](../docs/08-Web-UI-And-OpenShift.md#offline-build-wheelhouse).
The wheels are git-ignored; only this README is committed.

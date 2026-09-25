# Container image for the web UI (`meeting-assistant serve`).
#
# Built on Red Hat UBI so it runs unmodified under OpenShift's restricted SCC:
# the container runs as an arbitrary UID in group 0, so everything it writes
# lives under /opt/app-root, which the base image keeps group-writable.
#
# Corporate build networks (see docs/08-Web-UI-And-OpenShift.md):
#   --build-arg PIP_INDEX_URL=https://<mirror>/api/pypi/pypi/simple
#   --build-arg PIP_TRUSTED_HOST=<mirror host>   (only if TLS to it is intercepted)
#   --build-arg HTTPS_PROXY=http://<proxy>:<port>
# or put pre-downloaded wheels in wheelhouse/ to install fully offline.
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/python-312:latest
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST

WORKDIR /opt/app-root/src
COPY --chown=1001:0 pyproject.toml README.md LICENSE ./
COPY --chown=1001:0 src ./src
COPY --chown=1001:0 wheelhouse ./wheelhouse
COPY --chown=1001:0 deploy/container-entrypoint.sh /opt/app-root/bin/container-entrypoint.sh

RUN set -eu; \
    [ -n "${PIP_INDEX_URL:-}" ] || unset PIP_INDEX_URL; \
    [ -n "${PIP_TRUSTED_HOST:-}" ] || unset PIP_TRUSTED_HOST; \
    if ls wheelhouse/*.whl >/dev/null 2>&1; then \
        echo "Installing offline from wheelhouse/"; \
        pip install --no-cache-dir --no-index --find-links=wheelhouse . ; \
    else \
        pip install --no-cache-dir . ; \
    fi; \
    rm -rf wheelhouse build; \
    sed -i 's/\r$//' /opt/app-root/bin/container-entrypoint.sh; \
    chmod 0755 /opt/app-root/bin/container-entrypoint.sh; \
    mkdir -p /opt/app-root/data; \
    fix-permissions /opt/app-root -P

ENV MEETING_DATA_DIR=/opt/app-root/data \
    PYTHONUNBUFFERED=1

EXPOSE 8080
ENTRYPOINT ["/opt/app-root/bin/container-entrypoint.sh"]
CMD ["meeting-assistant", "serve", "--host", "0.0.0.0", "--port", "8080"]

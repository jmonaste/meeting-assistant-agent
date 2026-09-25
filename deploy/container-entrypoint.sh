#!/bin/sh
# Trust extra CA certificates before starting the app.
#
# Corporate endpoints and SSL-inspecting proxies present certificates signed
# by an internal CA that no public bundle contains. Mount those certificates
# (*.crt / *.pem) at $EXTRA_CA_DIR — e.g. from a ConfigMap — and they are
# appended to the system bundle, which SSL_CERT_FILE then points at. No root
# needed, so this works under OpenShift's arbitrary UID.
set -eu

EXTRA_CA_DIR="${EXTRA_CA_DIR:-/etc/meeting-assistant/ca}"
extra=""
for cert in "$EXTRA_CA_DIR"/*.crt "$EXTRA_CA_DIR"/*.pem; do
    [ -f "$cert" ] && extra="$extra $cert"
done

if [ -n "$extra" ]; then
    bundle="${TMPDIR:-/tmp}/ca-bundle.pem"
    # shellcheck disable=SC2086 # word splitting over the collected paths is intended
    cat "${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}" $extra > "$bundle"
    export SSL_CERT_FILE="$bundle" REQUESTS_CA_BUNDLE="$bundle"
    echo "Trusting extra CA certificates:$extra"
fi

exec "$@"

# 08 — Web UI and OpenShift deployment

The same pipeline as the CLI, behind a small browser UI (`meeting-assistant
serve`), packaged as a container image that runs on OpenShift and is reached
from your machine with `oc port-forward`. This chapter covers how the UI is
built, how to run it locally, and how to deploy it to an OpenShift namespace
step by step — including the corporate-network parts (package mirror, proxy,
internal CA).

## What the UI does

- **Upload** a transcript (`.vtt`, `.srt`, `.txt`, or paste the text), with
  optional context, report language and review depth; the endpoint-tuning
  knobs from [chapter 07](07-Endpoint-Tuning.md) are under *Endpoint tuning*.
- **Follow the run**: the phase stepper (ingest, plan, review passes,
  synthesis, gap-fill, compose), one progress bar per review pass with the
  running item count and how many items each pass added — the loop going dry
  is visible — plus endpoint retries (429/504) as they happen.
- **Read the report** rendered in the page, download the `.md` / `.json`,
  inspect the full run log (the same audit trail as `--log-file`) and the
  transcript.
- **Resume** a failed, interrupted or cancelled run from its last checkpoint,
  **cancel** a running one, **delete** old ones.
- **Settings** sets the endpoint URL, API key, models and every pipeline knob
  from the browser, saved on the server for all new analyses — no ConfigMap
  edit or pod restart. Each field shows whether its value comes from the UI,
  the deployment configuration or the code default, and can be reset.
- **Test connection** probes `GET {OPENAI_BASE_URL}/models` with exactly the
  TLS and proxy settings a run uses, and says *why* it failed (TLS / proxy /
  timeout / API key) — the first thing to press after deploying. Inside
  Settings it tests the values you are editing, before saving them.

## How it is built

```
browser ──HTTP polling──> FastAPI (web/app.py) ──> JobManager (web/jobs.py)
                                                     │  one worker thread
                                                     v
                                            run_pipeline()  (unchanged)
                                                     │
                                  <data dir>/jobs/<id>/  transcript, context.txt,
                                  job.json (status + live progress), run.log,
                                  .meeting_cache/ (checkpoints), out/*.md|json
```

Design decisions, and why:

| Decision | Why |
|----------|-----|
| A run is a **background job**, never a request | Runs take minutes to tens of minutes; a browser tab or a port-forward tunnel cannot be trusted to stay open that long. Closing the page or dropping the tunnel loses nothing. |
| **One job at a time** (queue + single worker) | The LLM endpoint is the bottleneck. `MEETING_MAX_CONCURRENCY` already parallelizes calls *within* a run; two runs at once would just double the 429s. |
| **Everything on disk** under the data dir | `job.json` is rewritten on every pipeline event, so a pod restart keeps the history; a job caught mid-run comes back as *interrupted* and **Resume** continues from the SQLite checkpoint (`run_pipeline(resume=True)`), so finished passes are not paid for again. |
| **Plain HTTP polling**, no websockets, no external JS/CSS/fonts | Works unchanged through `oc port-forward`, corporate proxies and browsers that cannot reach a CDN. The page is one static HTML file. |
| Report Markdown rendered **server-side with raw HTML disabled** | Model output is untrusted; `markdown-it-py` (already a dependency via `rich`) renders tables and headings but escapes any HTML the model might emit. |
| Pipeline progress via the existing `on_event` hook | Same node events the CLI's progress view consumes (`_Progress` mirrors `cli._ProgressView`); the pipeline itself is not modified. |
| Settings saved from the UI **layer over** the environment | Precedence per run: *New analysis* options > Settings (`<data dir>/settings.json`) > environment (ConfigMap/Secret, `.env`) > code defaults. The deployment can start with no configuration at all, and the environment still works for anyone who prefers it. Values are validated through the same `Settings` model before they are saved, so a typo is rejected instead of silently ignored. |
| The API key is **write-only** | The browser only ever gets "configured (…last 4)". It is stored in `settings.json` with mode 0600 on the pod's volume. |
| Container on **Red Hat UBI Python 3.12**, writable paths only under `/opt/app-root` | OpenShift's restricted SCC runs the container as a random UID in group 0; the UBI image and `fix-permissions` make that work without root. |

The UI has **no authentication**. It is meant to be reached through
`oc port-forward` (only people who can already `oc` into the namespace can
open it). Do not expose it with a public Route without putting an
authenticating proxy in front of it.

## Run it locally

```bash
pip install -e .                 # the web dependencies are part of the package
meeting-assistant serve          # http://127.0.0.1:8080, data in ./meeting-data
```

It reads the same `.env` as the CLI. Useful flags: `--port`, `--data-dir`
(or `MEETING_DATA_DIR`), `-v` to mirror info-level pipeline events to the
console. Pipeline warnings always reach the console; each job's full DEBUG
trail is in its own `run.log`.

## Container image

```bash
docker build -t meeting-assistant .        # or: podman build ...
docker run --rm -p 8080:8080 --env-file .env meeting-assistant
```

Behind a corporate network, pass what the build needs:

```bash
docker build -t meeting-assistant \
  --build-arg PIP_INDEX_URL=https://<artifactory>/api/pypi/pypi/simple \
  --build-arg PIP_TRUSTED_HOST=<artifactory host> \
  --build-arg HTTPS_PROXY=http://<proxy>:<port> .
```

`PIP_TRUSTED_HOST` is only needed when an SSL-inspecting proxy breaks TLS to
the mirror. If the build cannot reach any package index at all, see
[Offline build](#offline-build-wheelhouse).

The image runs `meeting-assistant serve --host 0.0.0.0 --port 8080`, keeps its
data in `/opt/app-root/data`, and its entrypoint appends any certificate
mounted in `/etc/meeting-assistant/ca/` (`*.crt`, `*.pem`) to the system CA
bundle and points `SSL_CERT_FILE` at the result — that is how the pod trusts
a corporate CA without root.

## Deploy to OpenShift

The manifests are in [`deploy/openshift/`](../deploy/openshift):

| File | What it creates |
|------|-----------------|
| `build.yaml` | `ImageStream` + `BuildConfig` (Docker strategy, binary source): the image is built **inside the cluster** from source you upload, so you need neither Docker on your laptop nor push access to a registry. |
| `app.yaml` | `PersistentVolumeClaim` (1 Gi, jobs and reports), `Deployment` (1 replica, probes, restricted security context, redeploys automatically on every new build), `Service` (port 8080, cluster-internal). |
| `config.env.example` | Template for the configuration ConfigMap. |

The commands below work the same in PowerShell, cmd and bash. Run them from
the repository root.

**1. Log in and select your namespace**

```bash
oc login --token=<token> --server=<api url>
oc project <your-namespace>
```

**2. Create the build and build the image**

```bash
oc apply -f deploy/openshift/build.yaml
oc start-build meeting-assistant --from-repo=. --follow
```

`--from-repo=.` uploads the **committed** state of your current branch (it runs
`git archive HEAD`), so a local `.env`, `.venv` or uncommitted change never
leaves your machine. Commit first if you want a change included. If the build
cannot download packages, uncomment the `buildArgs` in `build.yaml`
(`oc apply` it again) — or use the [offline build](#offline-build-wheelhouse).

**3. Configure the endpoint**

Two ways, and they combine:

- **From the UI** (simplest): skip to step 5, open the app, go to **Settings**
  and fill in *Base URL*, *API key* and the models; **Test connection**, then
  **Save**.
- **As deployment configuration**, for values that must exist before the
  process starts — the proxy variables and `LANGCHAIN_OPENAI_TCP_KEEPALIVE` —
  or if you prefer to keep everything in the namespace. Anything saved later
  in Settings overrides it.

```bash
cp deploy/openshift/config.env.example deploy/openshift/config.env   # Windows: copy ...
# edit config.env: OPENAI_BASE_URL, models, proxy ...
oc create configmap meeting-assistant-config --from-env-file=deploy/openshift/config.env
oc create secret generic meeting-assistant-secrets --from-literal=OPENAI_API_KEY=<key>
```

`config.env` is git-ignored — keep internal host names and proxies out of the
repository. The endpoint URL must be reachable **from the pod**, which is not
necessarily what works from your laptop. Do not copy `SSL_CERT_FILE` /
`REQUESTS_CA_BUNDLE` or any Windows path into it: step 4 handles certificates.

**4. (If needed) trust the corporate CA**

If the endpoint, or a proxy doing SSL inspection, uses a certificate from an
internal CA:

```bash
oc create configmap meeting-assistant-ca --from-file=corp-root-ca.crt
```

Alternatively, if the cluster administrators maintain a cluster-wide trusted
bundle, let OpenShift inject it — it already contains the corporate CAs:

```bash
oc create configmap meeting-assistant-ca
oc label configmap meeting-assistant-ca config.openshift.io/inject-trusted-cabundle=true
```

Either way it lands in `/etc/meeting-assistant/ca/` and the entrypoint picks
it up. Setting `MEETING_VERIFY_SSL=false` also works, but only as a last
resort for a trusted internal endpoint.

**5. Deploy**

```bash
oc apply -f deploy/openshift/app.yaml
oc rollout status deploy/meeting-assistant
```

No PVC quota in the namespace? In `app.yaml` replace the
`persistentVolumeClaim` of the `data` volume with `emptyDir: {}` and drop the
PVC document — the UI still works, but history is lost when the pod restarts.

**6. Open it**

```bash
oc port-forward svc/meeting-assistant 8080:8080
```

Browse to <http://localhost:8080> and press **Test connection** first.

### Day-2 operations

| Task | Command |
|------|---------|
| Deploy a new version of the code | commit, then `oc start-build meeting-assistant --from-repo=. --follow` (the Deployment rolls out by itself when the build finishes) |
| Change endpoint, key, models or tuning | **Settings** in the UI (applies to the next analysis) |
| Change the deployment configuration | edit `config.env`, then `oc create configmap meeting-assistant-config --from-env-file=deploy/openshift/config.env --dry-run=client -o yaml \| oc apply -f -` and `oc rollout restart deploy/meeting-assistant` |
| Follow the server log | `oc logs -f deploy/meeting-assistant` |
| Copy all reports to your laptop | `oc rsync <pod>:/opt/app-root/data/jobs ./jobs-backup` (`oc get pods -l app=meeting-assistant` for the pod name) |
| Remove everything | `oc delete all,pvc,configmap,secret -l app=meeting-assistant` plus the ConfigMaps/Secret you created by name |

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Build fails in `pip install` (timeouts, `Could not find a version`) | The build pod cannot reach PyPI. Set `PIP_INDEX_URL` (and `HTTPS_PROXY` if the mirror is outside) in `build.yaml`'s `buildArgs`, or build offline from a wheelhouse. |
| Build fails in `pip install` with `CERTIFICATE_VERIFY_FAILED` | SSL inspection between the build pod and the mirror: add `PIP_TRUSTED_HOST=<mirror host>`. |
| Build fails pulling `registry.access.redhat.com/ubi9/python-312` | The cluster cannot reach Red Hat's registry: uncomment the `from:` block in `build.yaml` to build on the cluster's own `openshift/python:3.12-ubi9` image. |
| Builds with the Docker strategy are not allowed in the namespace | Ask the platform team how images are meant to be built there (an internal registry you can push to, or a pipeline); the `Dockerfile` works with any of them. |
| Pod in `ImagePullBackOff` right after `oc apply -f app.yaml` | The image does not exist yet: wait for the build (`oc get builds`); the Deployment picks it up automatically. |
| **Test connection** → *TLS verification failed* | Step 4: mount the corporate CA. |
| **Test connection** → *timed out* / *proxy error* | The pod needs a proxy to reach the endpoint (`HTTPS_PROXY`), or the endpoint is internal and must be listed in `NO_PROXY`. |
| **Test connection** → *HTTP 401/403* | Wrong or missing `OPENAI_API_KEY` in the Secret. |
| Runs finish with 429/504 warnings | Endpoint tuning, exactly as for the CLI: [chapter 07](07-Endpoint-Tuning.md). Per run under *Endpoint tuning*, or for everyone in `config.env`. |
| Tunnel drops (`lost connection to pod`) | Normal for long idle `oc port-forward` sessions; re-run it. Running jobs are not affected. |

## Offline build (wheelhouse)

When the cluster build cannot reach any package index, install from wheels
you download on a machine that can — typically the laptop where the CLI
already works. From the repository root, in the virtualenv where
`pip install -e .` succeeded (with this version of the code installed):

```bash
pip freeze --exclude-editable > wheelhouse/requirements.txt
pip download -r wheelhouse/requirements.txt "setuptools>=61" --dest wheelhouse ^
    --only-binary=:all: --python-version 3.12 ^
    --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64
```

(`^` continues lines in cmd; use `` ` `` in PowerShell or `\` in bash.) This
fetches Linux wheels for Python 3.12 at the exact versions that work on your
machine. If it stops on a Windows-only package (for example `pywin32`), delete
that line from `wheelhouse/requirements.txt` and run it again. Then upload the committed code **plus** the wheels:

```bash
git archive -o build.tar HEAD
tar -rf build.tar wheelhouse
oc start-build meeting-assistant --from-archive=build.tar --follow
```

The Dockerfile sees `.whl` files in `wheelhouse/` and installs with
`--no-index`, touching no network. The wheels, `requirements.txt` and
`build.tar` are git-ignored.

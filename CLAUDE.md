# speech — CPU STT + TTS, one OpenAI-shaped surface

Four files are the service: `main.py` (routes), `stt.py` (faster-whisper),
`tts.py` (kokoro + ffmpeg for mp3), `transcript.py` (the growing transcript).
Nothing here authenticates: the ai plane (hanzoai/ai, behind api.hanzo.ai)
fronts this service and meters callers — including the anon grant tier (~3 uses
per anon) which lives THERE, not here.

## The streaming transcript (`/v1/audio/transcript`)

Batch stays at `/v1/audio/transcriptions`. The streaming sibling is a resource
that grows: `POST` to open, `POST` chunks to it, `DELETE` to close and flush.

- **Half duplex over plain HTTP, and this is forced, not preferred.** `ai` is a
  child process of the cloud pod reached over ZAP, and `zap-proto/http` builds a
  request as ONE frame — there is `FrameResponseHead` for streaming responses but
  no request-head frame, and `zip` contains no `Hijack`. So a WebSocket and a
  streamed request body BOTH cannot reach this service, no matter what the edge
  supports. `hanzoai/ingress` does pass upgrades, so testing `wss://` against
  ingress alone answers the wrong question — the wall is inside the pod.
- Audio is raw little-endian int16, mono, 16 kHz, in the body. No base64 (that
  is a WebSocket text-frame tax we do not pay), no container to demux.
- **Metering: `duration` on each ack is the seconds THAT push carried**, named as
  the batch route names it, so `ai` reads it into the same `AudioDurationSeconds`
  it already meters (`stt/openai.go:133`) with no new billing code. Summing the
  pushes yields the audio submitted once. It is arithmetic on bytes received, so
  it does not depend on the decoder returning anything.
- Whisper decodes a span, not a sample. The window is re-decoded as it fills;
  segments ending a guard-length before the audio are committed and their bytes
  freed, which is what bounds the window and stops committed text being revised.

**Sessions are pod-local, and `speech.yaml` sets `replicas: 2`.** Pushes reach
`speech.hanzo.svc`, which balances, so a session opened on one pod is not found
on the other. Whoever routes must pin a session to the pod that owns it; that
belongs in `ai` (which already does provider routing), not here — this service
decodes, it does not route. Until that exists the endpoint is not usable through
the plane, and `ai` has no route for it yet either.

- Weights bake at image BUILD (deterministic image, boots offline). Adding a
  model = extend `MODELS` in stt.py/tts.py AND the bake step in the Dockerfile.
- `whisper` / `kokoro` are the ROUTED names the ai plane's provider rows send;
  renaming them is a cross-repo change, not a local one.
- CI is `.hanzo/workflows/deploy.yml` (git.hanzo.ai act_runner; GitHub Actions
  is disabled account-wide). Version = pyproject's; a published tag is never
  overwritten — bump patch (x.x.x+1) per release. `pytest` gates the build and
  runs before the registry login, so a red suite cannot publish.
- **Two gates, and the second one exists because the first is blind.**
  `test_speech.py` replaces the MODEL (never `stt.segments` or `Transcript`), so
  it needs no weights and the window arithmetic is checkable exactly — its
  stand-in names each segment after the samples inside it, so a wrong trim
  changes the words, not just a length. But a suite that only ever sees a
  stand-in cannot tell a working decoder from a deleted one: breaking `stt.load`
  leaves it fully green. `test_live.py` closes that — kokoro speaks a sentence
  and whisper has to say it back, against real weights, and it kills that
  mutation. It runs as the `live` build stage, which sits above `final` so an
  untargeted `docker build` still produces an image without dev dependencies.
  Stub the MODEL, not the function: stubbing the function tests the route and
  leaves the real one free to regress.
- Deploy is declarative: universe `charts/app/values/hanzo/speech.yaml` pins
  the semver; no ingress (in-cluster only).
- **Verifying a deploy: the deployment's image is not the served image.** After a
  pin, `deploy/speech` reports the new tag immediately while the old pods still
  answer — the image is ~1.5 GB, so the pull alone runs about a minute and the
  ReplicaSets overlap. `readyReplicas` counts the OLD pods during that window, so
  it reads 2/2 and means nothing. Check for a READY pod on the new digest, then
  ask the running app what it serves (`/openapi.json`) or push real audio through
  it. There is no ingress, so probing is `kubectl port-forward svc/speech` — which
  attaches to ONE pod, so mid-rollout it may answer from either version.
- Pushes to main have twice failed to create an Actions run (the run is simply
  never created; `workflow_dispatch` on the same ref works, and other repos on
  the forge still trigger on push). Check for a run after pushing rather than
  assuming one exists — the previous release was dispatched by hand too.

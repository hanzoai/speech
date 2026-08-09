# speech — CPU STT + TTS, one OpenAI-shaped surface

Five files are the service: `main.py` (routes), `stt.py` (faster-whisper),
`tts.py` (kokoro + ffmpeg for mp3), `transcript.py` (the growing transcript),
`vad.py` (the squelch in front of it).
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
- **Metering: `duration` on each ack is the seconds THAT push put in front of a
  decoder**, named as the batch route names it, so `ai` reads it into the same
  `AudioDurationSeconds` it already meters (`stt/openai.go:133`) with no new
  billing code. It is arithmetic on the bytes that got past the squelch, so it
  does not depend on the decoder returning anything, and summing the pushes
  yields the audio decoded — never more than the audio submitted. A push the
  squelch drops reports 0.0; the push that OPENS it reports more than its own
  length, because it carries the lead-in that earlier pushes withheld.
- **Silence is not decoded and not billed** (`vad.py`). Silero already ships
  inside faster-whisper and already runs inside every decode, so running it in
  FRONT of the decoder costs no new weight and no new bake step — 0.67 ms per
  250 ms push, measured on the pod, against a decode of seconds. It is a squelch,
  not a per-window classifier: two thresholds to open and close, `HANG` of quiet
  before it closes, and `LEAD` of audio held back and handed over by the push
  that opens. Measured on the pod: an idle room bills 0.0 up to about -45 dBFS
  RMS of room tone (it starts to open on tone above that); a keyboard click and
  full-scale white noise do not open it; a meeting bills its speech plus about
  0.9 s per utterance and nothing for the lulls, so the share saved is 0% when
  someone is talking continuously and 86% at 30 s between turns.
  **The lead-in is the part that is easy to get wrong and expensive to lose.**
  Sliding a word across a push shows why: with `LEAD = 0`, 8 of 32 alignments
  clip up to 71.8 ms off the front of the word, because a push is admitted whole
  and a word beginning near the end of one has nothing in hand. At 0.5 s, none
  clip and the worst case still keeps 428 ms of margin. Hold the lead by slicing
  from the FRONT — `[-int(LEAD * SECOND):]` reads as the last N bytes and is the
  whole buffer when N is 0, which keeps every second of silence and hands the lot
  to the decoder on the next word.
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
- **There is no fast tier to add: `whisper` already IS the fast one.** Measured
  in the pod, both models on the same kokoro fixture, three runs each, median,
  no HTTP in the loop: `whisper` (distil-small.en) runs at 1.35–2.75× realtime
  and `whisper-small` (small) at 1.03–1.73×, so the default is 1.25–1.64× the
  faster of the two, and both transcribe the fixture at WER 0.00 — clean, at
  20 dB SNR and at 10 dB SNR alike. Splitting live captions from a final pass
  would mean a bigger model for the final one, and `small` at 1.03× is already
  the largest that keeps up with a meeting on four cores with no GPU in the
  cluster. `whisper-small` earns its place on LANGUAGE, not on accuracy or
  speed: distil-small.en is English-only.
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

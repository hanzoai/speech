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

## What it sustains, and where that number comes from

`bench.py` is the instrument: `say` once for a corpus of real speech carrying a
word-to-time map, then `cost` for what a decode pass costs and `talk` to drive N
sessions in real time. Everything below was measured with it **on a four vCPU
DigitalOcean worker** — the shape a speech pod runs in. That qualifier is the
whole point: the same code and the same run on four threads of a desktop Zen 5
retires 13.7 audio-seconds per wall-second where the pod retires 5.7. A capacity
number measured on the box is 2.4x the truth.

**Ack latency cannot see any of this and none of these numbers is one.** A push
returns whatever text the session has, so it costs a memcpy whether the decoder
is idle or minutes behind — 3.8 ms while the backlog grew fourfold. The metric is
backlog, as lag: wall seconds elapsed minus seconds of audio already transcribed.
The verdict is its SLOPE over the second half of a run, in seconds of lag added
per minute of meeting. Runs must be MINUTES long: a stalling session looks fine
for its first four minutes, which is how seconds-long runs came to predict eight
concurrent meetings and were wrong by an order of magnitude.

**A pass costs what it costs, almost regardless of what it covers.** Whisper
always encodes a thirty second mel window, so on the pod one pass takes 2.7 s over
a one second window and 3.2 s over an eight second one — about 8.4 CPU-seconds
either way. Capacity is a budget of PASSES, not of audio, and the way to buy it is
to pass less often, never to pass over less.

**Past the mel window that cost multiplies.** 24 s -> 5.0 s per pass, 30 s -> 7.5,
45 s -> 23.5. It is a cliff, not a slope.

**So the pod has a throughput PEAK, and the peak is the instability.** In steady
state a session gets one pass per interval and that pass covers the audio that
arrived during it, so audio retired per wall second IS the number of real-time
streams served, and lag is roughly the window. Measured on the pod:

    window            8 s   12 s   16 s   24 s   30 s
    streams served    3.1    4.4    5.0    3.9    2.8

Below the peak the loop is self-correcting: a longer window is cheaper per second,
so lag comes back. Past it, a longer window is DEARER per second, so lag grows,
which lengthens the window — runaway. That is why the failure is a cliff and not a
gentle degradation, and why the working point has to sit on the rising branch
rather than at the top of the curve.

**Measured, in eight minute runs at 250 ms chunks**, and the two agree to within a
few percent — a session settles where its window buys exactly its own concurrency:

    sessions   lag       slope         verdict
    3          7.7 s     -2.36 s/min   settles
    4          12.7 s    -0.45 s/min   settles

So: **four concurrent streams of continuous speech per four vCPU pod**, with the
transcript about thirteen seconds behind, or three at eight seconds. Two replicas
is eight. The curve says five sits ON the peak and six has no fixed point at all.

The failure past that edge is not a gentle one. The same runs against the earlier
two-worker build, which peaked lower, settled at one and two sessions and at FOUR
ran away to 326 s of lag at +44 s/min — and closing those sessions then blocked for
400 s each, because `DELETE` decodes everything that piled up.

**A session's ceiling is `LIMIT` = 600 s, so a 45 minute meeting is five sessions,
not one.** Closing decodes the whole remaining window and blocks, and the transport
admits one outstanding chunk, so the roll costs the microphone that whole pause.
At three streams that is a couple of seconds every ten minutes; on a session that
has fallen behind it is minutes.

### What more of it costs

Per pod, three streams. Per stream of continuous speech that is 1.33 vCPU. So:

    concurrent streams     pods (4 vCPU)     vCPU
    10                     4                 16
    100                    34                136
    1000                   334               1336

Per-participant streams are nearly free next to that, because the squelch means a
quiet participant is never decoded: about 0.27% of a core for the Silero pass on
250 ms of quiet. A thousand five-person meetings is ~1000 voiced streams plus
~13 cores of squelch, not 5000 streams.

Buying 334 four-core nodes is the wrong answer, so state the alternatives with the
number each has to beat.

**Batching is not the lever, and that is measured, not assumed.** The fixed thirty
second encode looks exactly like the shape that amortizes, and faster-whisper
carries `BatchedInferencePipeline`, but on CPU it does not amortize: eight 12 s
windows took 11.64 s sequentially and 6.79 s batched — 1.71x the wall clock for
**the same CPU** (24.1 vs 23.6 CPU-seconds). Batching overlaps the work, it does
not reduce it, so it helps a pod whose cores are idle and does nothing for one
that is already saturated. Where it would pay is a GPU, where the fixed cost is
underused hardware rather than arithmetic.

A GPU has to retire 1000 audio-seconds per wall second of distil-small.en over
~16 s windows to replace those 334 pods. **No GPU figure here is measured, the
cluster has no GPU node, and every large node is tainted for CI**, so that is a
procurement decision to be measured on the card and not believed before it is.

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

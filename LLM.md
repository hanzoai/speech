# speech — Hanzo voice on CPU, one OpenAI-shaped surface

One Go binary over sherpa-onnx (k2-fsa, C++ on ONNX Runtime, through cgo).
It is its own service, not code in the cloud binary, because cloud builds with
`CGO_ENABLED=0` — the gateway routes to it at `http://speech.hanzo.svc/v1`.

Nothing here authenticates: the ai plane (hanzoai/ai, behind api.hanzo.ai)
fronts it, authenticates every caller and meters what this service reports.
Customer-facing names say "Hanzo voice"; the model ids below are upstream ids
behind the gateway (`zen-scribe`, `zen-voice-mini` route to them).

## Files

`main.go` boot · `http.go` routes · `speech.go` routing + batch transcription ·
`transcript.go` the growing transcript + squelch · `audio.go` ffmpeg in, wav/pcm
out · `voices.go` · `language.go` · `weights.go` S3 fetch · `sherpa.go` the ONLY
file that touches the native runtime. `speech_test.go` replaces every model;
`live_test.go` runs the real ones.

## Models and licences

| id | what | licence | source |
|---|---|---|---|
| `parakeet` | NVIDIA Parakeet TDT 0.6B v3, int8, 25 European languages | CC BY 4.0 | huggingface.co/nvidia/parakeet-tdt-0.6b-v3, sherpa-onnx int8 export |
| `whisper` | OpenAI Whisper large-v3-turbo, int8, any language | MIT | github.com/openai/whisper; encoder from sherpa-onnx-whisper-turbo, decoder re-exported (below) |
| (LID) | Whisper tiny int8, spoken language identification | MIT | sherpa-onnx-whisper-tiny |
| `kokoro` | Kokoro-82M v1.0, 54 voices | Apache-2.0 | huggingface.co/hexgrad/Kokoro-82M, sherpa-onnx kokoro-multi-lang-v1_0 |
| (VAD) | Silero VAD | MIT | github.com/snakers4/silero-vad |

espeak-ng (GPL-3.0+) is built into sherpa-onnx's TTS front end and its data
ships with the Kokoro weights. NOTICE carries the full attributions. Parakeet's
CC BY 4.0 needs this credit wherever Hanzo voice is described:

> Hanzo voice speech recognition uses NVIDIA Parakeet TDT 0.6B v3
> (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, licensed under
> CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The model was
> converted to ONNX and quantized to 8-bit integers by the k2-fsa sherpa-onnx
> project.

**`whisper` keeps answering `whisper`**: zen-scribe and the voice service route
to that id. **Pocket TTS is not served**: the Go API is clean (OfflineTts +
reference audio), but sherpa-onnx 1.13.8's int8 export dropped the opening
words of the test sentence at default settings, came back empty for one voice,
was exact for another only at 8 flow steps and temperature 0.4, and ran at RTF
1.5–3.7 on four threads (0.5 at eight); its README also says non-commercial
against its own CC BY 4.0 LICENSE file.

## Routing

`max_seconds` on a transcription holds the caller to a length: the ffmpeg
decode stops a quarter second past it and audio that runs over is refused 413
before the model runs. The ai plane's public lane (visitors with no account)
sends it; a signed-in caller's request carries none.

`whisper` decodes everything, in the language named or the one it detects.
`parakeet` decodes its 25 languages itself and hands the rest to whisper: a
`language` outside its set goes straight there; with none named, the speech
(silences cut, at most 30 s) is identified by the tiny whisper first. The
fallback hands whisper `""` when the language was identified, so the turbo
model detects it again — forty times the identifier's size, one decoder step.

`language` takes ISO-639-1, with or without a region (`en-US`, `pt_BR`). A code
whisper does not know is a 400 HERE: handed to sherpa it calls `exit(-1)`. So
does a VAD window of the wrong size; `sherpa.go` validates before it calls, and
never takes `&samples[0]` of an empty slice.

whisper's language is recognizer config, not stream config (1.13.8 reads no
per-stream option for whisper), so each recognizer has one worker goroutine
that sets the language and decodes in the same turn. One call at a time per
model, each spread over every core (`GOMAXPROCS`, i.e. the pod's CPU limit);
an unbuffered channel queues callers FIFO, and that queue is the fairness.

## Cutting audio into decodes

Every decode is a span of speech that silero found, plus `Pad` (0.5 s) either
side — but never past halfway to the neighbouring span (`reach`), so no word is
heard twice and no other speaker's tail rides in. Where a span starts at the
first sample, the lead-in is silence (`excerpt` zero-fills): silero starts a
span late on a soft onset, and with 0.25 s of pad "The quick brown fox" came
back as "Quick brown fox" from both models.

- **whisper merges** neighbouring spans closer than 2 s, up to 20 s, for context
  and to pay its fixed ~0.55 s per decode once. Nothing is ever longer than 28 s
  (its ceiling is 30).
- **Parakeet decodes span by span, and spans end at 0.3 s of quiet.** Measured
  on its int8 export: two speakers with room tone between them, decoded as one
  span, lost the second speaker in 24 of 64 trials. Same voice twice never
  failed, digital silence between them never failed; it is a speaker change
  across noise. Span by span at a 0.5 s pause it was 2 of 40 (both 0.3 s turns);
  at 0.3 s, 0 of 40, and 0 of 32 through the growing transcript.

Segments in verbose_json are those spans (id, seek, start, end, text — the
fields this decoder actually knows). Words come from token timestamps: Parakeet
is a transducer and times every token; whisper is timed by DTW over its
alignment heads, which needs the decoder below.

**The whisper decoder is our export.** The published sherpa turbo decoder has
three outputs and times nothing. `scripts/whisper/export-onnx-with-attention.py`
(sherpa-onnx v1.13.8) adds `cross_attention_weights`; we ran it with the fp32
encoder export cut out (identical to the published one), quantized the decoder
to int8 (MatMul), and stamped the alignment-head metadata (`version: 2`,
`alignment_heads: 2:4,2:11,3:3,3:6,3:11,3:14`, `n_alignment_heads: 6`) onto the
published int8 encoder. Needs `torch==2.5.1`, `openai-whisper==20250625` built
with `setuptools<70`, `onnx==1.17.0`, `onnxruntime==1.20.1`.

## Kokoro

Voices by Kokoro id (`af_heart` default, `bf_emma`, `zf_xiaoxiao`, …) and by
OpenAI name (alloy→af_alloy, ash→am_michael, ballad→bm_george, coral→af_sarah,
echo→am_echo, fable→bm_fable, nova→af_nova, onyx→am_onyx, sage→af_kore,
shimmer→af_bella, verse→am_puck). Unknown voice: 400 naming every valid one.
`speed` 0.25–4.0. Formats are encoded, never substituted: mp3 audio/mpeg, wav
audio/wav, opus audio/ogg, aac audio/aac, flac audio/flac, pcm audio/pcm (raw
s16le 24 kHz). wav and pcm are written in Go; the rest go through ffmpeg.

The language comes from the voice's first letter, as an **espeak-ng voice file
name**: sherpa passes it to `espeak_SetVoiceByName`, which matches file names,
so Kokoro's own `en-gb` and `fr-fr` find nothing and the request fails to
tokenize. British is `en`, French `fr`. Non-Han text always goes through
espeak in that language (sherpa takes that path whenever a language is set,
and the model's default is en-us), so only the Chinese lexicon ships. espeak
reads no kanji: Japanese voices speak kana correctly and kanji badly, and Hindi
round-trips poorly. European voices round-trip exactly.

## The growing transcript (`/v1/audio/transcript`)

`POST` opens (`{model, language, format: pcm16, rate: 16000, channels: 1}` →
201 with `id`, `at`, `chunk_ms` 256, `max_bytes` 65536, `max_seconds` 600,
`idle_seconds` 30), `POST /{id}` pushes raw little-endian int16 mono 16 kHz,
`DELETE /{id}` decodes the rest and commits it. The ai plane
(`controllers/zap_transcript.go`) and the voice service both drive it.

- **`at`** is this pod (`POD_IP`); sessions live in one process's memory and the
  plane pins a session to the pod that opened it. Empty without `POD_IP`.
- **Metering**: `duration` on an ack is the seconds THAT push put in front of a
  decoder; `seconds` is the session's running total, which is what the plane
  bills (delta of `seconds`). Both are arithmetic on admitted samples, never on
  what a decoder returned. Close bills nothing.
- **Silence is not decoded and not billed.** A per-session silero detector is
  the squelch: it opens on two windows above 0.5, holds through anything above
  0.35, closes after 0.8 s of quiet — silero's own state machine, the same
  thresholds the Python squelch ran. A push is admitted whole or not at all;
  0.5 s of `Lead` held while quiet is handed over by the push that opens. An
  empty room of -45 dBFS room tone bills 0 (live test).
- **The window**: a pass finds the spans in it; spans that end a `Guard` (1 s)
  before its end are decoded on their own and committed, their audio freed; the
  rest is decoded as one piece and becomes `pending`. Past 12 s the window is
  committed inside unbroken speech, so it cannot grow, and the cut goes in the
  quietest 80 ms of the 3 s before the guard (`quietest`), never at the newest
  sample: cut mid-word, both decodes answer with the word ("music to to
  children"). Whisper's 28 s split takes its lull the same way. One pass per
  session at a time; passes queue in the recognizer's FIFO behind other sessions'.
- A parakeet session that names no language waits for 2 s of speech, identifies
  it once, and keeps the answer.

**Streaming talk mode is `/v1/voice`, not a socket here.** The Python LLM.md
said a WebSocket could not cross the plane (zap-proto/http built a request as
one frame, zip had no Hijack). That no longer holds: zip v1.37.25
(`transport.go`) carries an upgrade through to the plugin with Hijack, and
hanzoai/ai serves `/v1/voice` (hanzoai/voice: the OpenAI realtime protocol)
behind it, live (`GET https://api.hanzo.ai/v1/voice/health` → 200). That
socket ends in the ai plane, which calls this service over plain HTTP: the
growing transcript to listen, `/v1/audio/speech` (wav) to speak. So this
service still needs no socket of its own; the voice service's `VOICE_STT`
picks which model listens.

## Weights

Never in git, never in the image. `weights.sum` (sha256, size, path) is
compiled in and names every file; at boot the service fetches what is missing
from `s3://oca95588b94408293-models/speech/<path>` (the org's private `models`
bucket on hanzoai/s3, `http://s3.hanzo.svc:9000`, SigV4 with the store key from
KMS `hanzo/prod:/hanzo-s3`, synced as secret `s3-credentials`) into an emptyDir
at `/models`, hashing each as it streams and renaming it into place only when
it agrees. `/healthz` is 503 until every model has loaded. A file already there
at the right size is kept, so a dev box with `MODELS=/home/z/models/speech`
fetches nothing. 371 files, 2.19 GB.

Changing a weight: put the file under the models dir, upload it to the same
key, regenerate `weights.sum`:

    cd $MODELS && find . -path ./.staging -prune -o -type f -print | sed 's|^\./||' | LC_ALL=C sort |
      while IFS= read -r f; do printf '%s %s %s\n' "$(sha256sum "$f" | cut -c1-64)" "$(stat -c %s "$f")" "$f"; done

## Tests

`go test ./...` — the unit suite replaces the models (an ear that names the
tones it is handed, an amplitude VAD running the squelch's window arithmetic),
so window cuts, billing, routing and every HTTP shape are checked exactly with
no weights. The live tests run when `MODELS` holds every file in weights.sum
and skip otherwise: kokoro speaks and parakeet and whisper must say it back at
word error 0, through every browser container (webm, ogg, m4a, mp3, wav);
Chinese and Hindi make parakeet fall back to whisper; a growing transcript, an
empty room, a two-speaker meeting. ffmpeg must be on PATH (on dgx,
`~/.local/bin/ffmpeg` is an x86 static binary under emulation; put
`/usr/bin` first).

## Measured

**In the pod** (AMD EPYC 9R14, 4-core limit, GOMAXPROCS 4), wall time per
request from the service's own log (ffmpeg decode and VAD included), 10 s of
speech: parakeet RTF 0.055, whisper 0.31–0.33, kokoro 0.25–0.28; on 5 s,
parakeet 0.09–0.10 and whisper 0.39 (whisper pays ~0.55 s per decode before it
reads a second of audio). Memory: RSS 3.51 GB at rest with every model loaded,
3.95 GB peak with parakeet, whisper and kokoro decoding at once; kubelet working
set 3866Mi, so the chart requests 4.5Gi. A cold pod fetches all 371 files from
S3 in 17 s and loads every model in 2.2 s: Ready 19 s after start.

On dgx, four big cores (`taskset -c 5-8`): parakeet RTF 0.024, whisper 0.25,
kokoro 0.21, language ID 0.14 s per call.

## Release and deploy

No CI lane runs this repo: GitHub reads no `.hanzo/` directory and the forge
copy of a hanzoai repo is a mirror that runs nothing. A release goes through
the build door, both architectures natively:

    curl -X POST https://api.hanzo.ai/v1/build -H "Authorization: Bearer $(hanzo auth token)" \
      -d '{"repo":"https://github.com/hanzoai/speech.git","sha":"<commit>","image":"ghcr.io/hanzoai/speech:<x.y.z>",
           "dockerfile":"Dockerfile","context":".","platforms":["linux/amd64","linux/arm64"]}'

The verdict is `GET https://api.hanzo.ai/v1/platform/builds`. Bump the patch,
never overwrite a published tag. Deploy is universe
`charts/app/values/hanzo/speech.yaml` (tag + digest, `cd.automated`); the AWS
overlay `infra/aws/values/hanzo/speech.yaml` places it on the apps node.
After a pin, check a READY pod on the new digest and push real audio through
it (`kubectl port-forward` attaches to one pod).

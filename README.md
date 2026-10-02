# speech

Hanzo voice on CPU: speech-to-text and text-to-speech behind one OpenAI-shaped
surface. One Go binary over [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Routes

- `POST /v1/audio/transcriptions` — multipart `file` + `model` (`parakeet` |
  `whisper`) [+ `language`, `response_format` json | text | verbose_json,
  `timestamp_granularities[]` word | segment]. Any container a browser records.
- `POST /v1/audio/speech` — `{model: kokoro, input, voice, response_format,
  speed}` → mp3 (default), wav, opus, aac, flac or pcm, labelled as what it is.
- `POST /v1/audio/transcript`, `POST|DELETE /v1/audio/transcript/{id}` — a
  transcript that grows as raw pcm16 arrives.
- `GET /v1/models`, `GET /healthz` (ready once every model has loaded).

## Trust model

No auth of its own, in-cluster only. `api.hanzo.ai` fronts it through the ai
plane, which authenticates and meters every caller; a provider row pointing at
`http://speech.hanzo.svc/v1` is the whole integration.

## Run

    MODELS=/path/to/weights go run .

`weights.sum` names every weight file. Set `S3_ENDPOINT`, `S3_BUCKET` and an
S3 key to fetch the missing ones at boot; see LLM.md.

# speech

CPU speech for the Hanzo platform: **STT** (faster-whisper, CTranslate2 int8)
and **TTS** (kokoro-onnx, 82M) behind one OpenAI-shaped surface.

## Routes

- `POST /v1/audio/transcriptions` — multipart `file` + `model` (+`language`,
  `response_format`) → `{"text": …}`
- `POST /v1/audio/speech` — `{model, input, voice, response_format}` → audio
  bytes (mp3 default, wav native)
- `GET /v1/models`, `GET /healthz`

## Trust model

No auth of its own, in-cluster only. `api.hanzo.ai` fronts it through the ai
plane, which authenticates and meters every caller — a provider row pointing at
`http://speech.hanzo.svc/v1` is the whole integration. Weights bake into the
image at build; the container boots without the network.

## Run

    uv sync && uv run uvicorn main:app --port 8000

Phase 2 is `/v1/realtime`: websocket full-duplex (silero-vad segmentation,
partial transcripts, kokoro frames, barge-in) plus camera-frame relay into the
multimodal models.

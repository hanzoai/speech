# speech — CPU STT + TTS, one OpenAI-shaped surface

Three files are the service: `main.py` (routes), `stt.py` (faster-whisper),
`tts.py` (kokoro + ffmpeg for mp3). Nothing here authenticates: the ai plane
(hanzoai/ai, behind api.hanzo.ai) fronts this service and meters callers —
including the anon grant tier (~3 uses per anon) which lives THERE, not here.

- Weights bake at image BUILD (deterministic image, boots offline). Adding a
  model = extend `MODELS` in stt.py/tts.py AND the bake step in the Dockerfile.
- `whisper` / `kokoro` are the ROUTED names the ai plane's provider rows send;
  renaming them is a cross-repo change, not a local one.
- CI is `.hanzo/workflows/deploy.yml` (git.hanzo.ai act_runner; GitHub Actions
  is disabled account-wide). Version = pyproject's; a published tag is never
  overwritten — bump patch (x.x.x+1) per release. `pytest` gates the build and
  runs before the registry login, so a red suite cannot publish.
- `test_speech.py` stubs the two model calls, so it needs no weights: what is
  tested is this service's CONTRACT — the duration it returns (the ai plane
  meters transcription per minute) and the container it claims (a format is
  encoded or refused, never substituted). Both are where its two shipped
  defects lived. Test `stt.transcribe`/`tts.speak` by stubbing the MODEL, not
  the function: stubbing the function tests the route and leaves the real one
  free to regress.
- Deploy is declarative: universe `charts/app/values/hanzo/speech.yaml` pins
  the semver; no ingress (in-cluster only).

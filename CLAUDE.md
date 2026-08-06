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
  overwritten — bump patch (x.x.x+1) per release.
- Deploy is declarative: universe `charts/app/values/hanzo/speech.yaml` pins
  the semver; no ingress (in-cluster only).

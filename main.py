"""One OpenAI-shaped surface over CPU speech: whisper in, kokoro out.

No auth of its own — in-cluster only. The ai plane (api.hanzo.ai) authenticates
and meters every caller before a request reaches this service; a provider row
pointing at http://speech.hanzo.svc/v1 is the whole integration.
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

import stt
import tts

app = FastAPI(title="speech", docs_url=None, redoc_url=None)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/models")
def models():
    names = (*stt.MODELS, *tts.MODELS)
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "hanzo"} for n in names]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    if model not in stt.MODELS:
        raise HTTPException(404, f"unknown model {model!r}")
    text = stt.transcribe(model, await file.read(), language)
    if response_format == "text":
        return Response(text, media_type="text/plain")
    return JSONResponse({"text": text})


@app.post("/v1/audio/speech")
def speech(ask: tts.Ask):
    if ask.model not in tts.MODELS:
        raise HTTPException(404, f"unknown model {ask.model!r}")
    if ask.voice not in tts.VOICES:
        raise HTTPException(400, f"unknown voice {ask.voice!r}")
    audio, mime = tts.speak(ask)
    return Response(audio, media_type=mime)

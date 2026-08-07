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
    text, duration = stt.transcribe(model, await file.read(), language)
    if response_format == "text":
        return Response(text, media_type="text/plain")
    # verbose_json carries `duration`, exactly as OpenAI's does — and the ai
    # plane asks for it because that duration is what meters the call. Plain
    # json stays {"text"} so a client reading the standard shape is unaffected.
    if response_format == "verbose_json":
        return JSONResponse({"text": text, "duration": duration})
    return JSONResponse({"text": text})


@app.post("/v1/audio/speech")
def speech(ask: tts.Ask):
    if ask.model not in tts.MODELS:
        raise HTTPException(404, f"unknown model {ask.model!r}")
    if ask.voice not in tts.VOICES:
        raise HTTPException(400, f"unknown voice {ask.voice!r}")
    # Refuse a format we cannot make, rather than answering in a different one:
    # every format below is really encoded, and anything else is a 400 naming
    # what IS available. Substituting silently is how a caller asking for opus
    # received an MP3 labelled audio/opus.
    if ask.response_format not in tts.FORMATS:
        raise HTTPException(
            400,
            f"unsupported response_format {ask.response_format!r}; "
            f"supported: {', '.join(sorted(tts.FORMATS))}",
        )
    audio, mime = tts.speak(ask)
    return Response(audio, media_type=mime)

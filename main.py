"""One OpenAI-shaped surface over CPU speech: whisper in, kokoro out.

No auth of its own — in-cluster only. The ai plane (api.hanzo.ai) authenticates
and meters every caller before a request reaches this service; a provider row
pointing at http://speech.hanzo.svc/v1 is the whole integration.
"""

import os
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import stt
import transcript
import tts

app = FastAPI(title="speech", docs_url=None, redoc_url=None)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/models")
def models():
    names = (*stt.MODELS, *tts.MODELS)
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "hanzo"} for n in names]}


# What a caller may ask to be timed, and OpenAI's two names for it.
GRANULARITIES = {"word", "segment"}


def granularities(form) -> set[str]:
    """`timestamp_granularities`, either spelling.

    OpenAI's own SDKs encode a multipart array with a bracketed name and
    hand-rolled clients almost always send the bare one. Both mean the same
    thing, so both are read here — in ONE place — rather than leaving half the
    clients silently untimed.
    """
    asked = {*form.getlist("timestamp_granularities[]"), *form.getlist("timestamp_granularities")}
    unknown = asked - GRANULARITIES
    if unknown:
        raise HTTPException(
            400,
            f"unknown timestamp_granularities {sorted(unknown)}; "
            f"supported: {', '.join(sorted(GRANULARITIES))}",
        )
    return asked


def timed_word(w) -> dict:
    return {"word": w.word.strip(), "start": w.start, "end": w.end}


def timed_segment(s) -> dict:
    return {
        "id": s.id,
        "seek": s.seek,
        "start": s.start,
        "end": s.end,
        "text": s.text.strip(),
        "tokens": list(s.tokens),
        "temperature": s.temperature,
        "avg_logprob": s.avg_logprob,
        "compression_ratio": s.compression_ratio,
        "no_speech_prob": s.no_speech_prob,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    if model not in stt.MODELS:
        raise HTTPException(404, f"unknown model {model!r}")
    asked = granularities(await request.form())
    # Timings ride the verbose body and nowhere else, so asking for them
    # alongside a body that cannot carry them is a mistake worth naming: the
    # alternative is charging for the alignment pass and then discarding it.
    if asked and response_format != "verbose_json":
        raise HTTPException(400, "timestamp_granularities requires response_format=verbose_json")
    want = asked or {"segment"}  # OpenAI's default granularity
    text, duration, heard = stt.transcribe(model, await file.read(), language, words="word" in want)
    if response_format == "text":
        return Response(text, media_type="text/plain")
    # Plain json stays {"text"} so a client reading the standard shape is
    # unaffected.
    if response_format != "verbose_json":
        return JSONResponse({"text": text})
    # verbose_json carries `duration`, exactly as OpenAI's does — and the ai
    # plane asks for it because that duration is what meters the call. It
    # carries the timings too: the decode already knew where every segment
    # started and stopped, and dropping that on the floor is what left every
    # caption in the estate cutting on lines it had to invent word boundaries
    # inside.
    body = {"task": "transcribe", "duration": duration, "text": text}
    if "segment" in want:
        body["segments"] = [timed_segment(s) for s in heard]
    if "word" in want:
        body["words"] = [timed_word(w) for s in heard for w in (s.words or ())]
    return JSONResponse(body)


class Open(BaseModel):
    model: str = "whisper"
    language: str | None = None
    format: str = "pcm16"
    rate: int = stt.RATE
    channels: int = 1


# Where THIS process answers. A growing transcript is a window held in one
# process's memory, so every push after the first has exactly one server that can
# take it — and a Service address round-robins per connection, which sends half of
# them to a replica that has never heard of the session.
#
# So `open` says where it lives and the caller addresses that pod for the rest of
# the session. Empty when POD_IP is unset, which is the honest answer for a single
# process: there is nothing to pin to, and the caller keeps the address it has.
# The deployment sets it from the downward API; with more than one replica it is
# not optional, and TestOpenNamesWhereItLives is what says so.
HERE = f"http://{os.environ['POD_IP']}:{os.getenv('PORT', '8000')}" if os.getenv("POD_IP") else ""


# ── the streaming sibling of /v1/audio/transcriptions ───────────────────────
#
# Half duplex over ordinary HTTP: the client POSTs a chunk of audio and the
# RESPONSE to that POST carries the transcript. No socket, no second leg.
#
# That is not a preference. `ai` is a child process reached over ZAP, and ZAP
# builds a request as ONE frame — there is no request-head frame and no upgrade
# path, so neither a WebSocket nor a streamed request body can reach this service.
# A chunk small enough to fit one frame is the shape the transport actually has.
# It also makes backpressure structural: chunk n+1 cannot be sent until n is
# acked, so a slow server slows the microphone instead of queueing audio.


@app.post("/v1/audio/transcript", status_code=201)
def transcript_open(ask: Open):
    if ask.model not in stt.MODELS:
        raise HTTPException(404, f"unknown model {ask.model!r}")
    # Raw audio carries no header saying what it is, so a mismatch here is silent
    # noise rather than an error: 8 kHz read as 16 kHz transcribes as gibberish.
    # Refuse it by name instead of resampling something we were told is correct.
    if (ask.format, ask.rate, ask.channels) != ("pcm16", stt.RATE, 1):
        raise HTTPException(
            400,
            f"expected pcm16 mono at {stt.RATE} Hz; "
            f"got {ask.format!r} {ask.channels}ch at {ask.rate} Hz",
        )
    live = transcript.begin(ask.model, ask.language)
    return {
        "id": live.id,
        "at": HERE,
        "model": live.model,
        "format": "pcm16",
        "rate": stt.RATE,
        "channels": 1,
        "chunk_ms": round(transcript.CHUNK / stt.SECOND * 1000),
        "max_bytes": transcript.CEILING,
        "max_seconds": transcript.LIMIT,
        "idle_seconds": transcript.IDLE,
        "expires_at": int(time.time() + transcript.IDLE),
    }


@app.post("/v1/audio/transcript/{tid}")
async def transcript_push(tid: str, request: Request):
    """Take a chunk, answer with the newest state. `duration` is what this push
    consumed — the field the plane meters, named as the batch route names it."""
    live = transcript.find(tid)
    if live is None:
        raise HTTPException(404, f"no transcript {tid!r}")
    pcm = await request.body()
    if len(pcm) > transcript.CEILING:
        raise HTTPException(413, f"chunk is {len(pcm)} bytes; limit {transcript.CEILING}")
    if len(pcm) % stt.WIDTH:
        raise HTTPException(400, f"{len(pcm)} bytes is not whole int16 frames")
    if live.full:
        raise HTTPException(409, f"transcript is at its {transcript.LIMIT}s limit; close it")
    return live.state(live.push(pcm))


@app.delete("/v1/audio/transcript/{tid}")
def transcript_close(tid: str):
    """Close: decode the remainder and commit it. No audio arrives, so nothing is
    billed here — every second was already metered by the push that carried it."""
    live = transcript.drop(tid)
    if live is None:
        raise HTTPException(404, f"no transcript {tid!r}")
    live.close()
    return live.state()


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

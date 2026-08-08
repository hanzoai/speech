"""faster-whisper (CTranslate2, int8) — the STT half.

Weights bake into the image at build, so loading here is a read from the local
cache: lazy, once per process, kept for its life.
"""

import io
from functools import lru_cache

import numpy
from faster_whisper import WhisperModel

# Request `model` → weight. `whisper` is the routed name the ai plane sends;
# distil covers English fast, `whisper-small` is the multilingual step up.
MODELS = {
    "whisper": "distil-small.en",
    "whisper-small": "small",
}

# Whisper's native rate, and the one faster_whisper.audio resamples everything to
# (audio.py: sampling_rate = 16000). Raw audio arrives already at this rate, mono,
# little-endian int16 — so there is nothing to resample and nothing to demux.
RATE = 16000
WIDTH = 2
SECOND = RATE * WIDTH  # bytes of raw audio per second


@lru_cache(maxsize=None)
def load(model: str) -> WhisperModel:
    return WhisperModel(MODELS[model], device="cpu", compute_type="int8")


def samples(pcm: bytes) -> numpy.ndarray:
    """Raw little-endian int16 → the float32 in [-1, 1) the model consumes.

    Handing the model an array skips faster-whisper's demux entirely: a container
    goes through ffmpeg, an array does not. Raw audio has no container, so this is
    the shorter path AND the exact one.
    """
    return numpy.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0


def _hear(model: str, audio, language: str | None):
    """The one call into the model. Both entries below differ only in what they
    hand it — a container to demux, or samples already in hand."""
    return load(model).transcribe(audio, language=language, vad_filter=True)


def segments(model: str, pcm: bytes, language: str | None) -> list:
    """Decode raw audio, returning the model's segments WITH their timings.

    A growing transcript needs the timings, not just the text: a segment that
    ended well before the audio does is settled and can be committed, while the
    tail is still moving. `transcribe` throws the timings away because a batch
    caller has nothing to do with them.
    """
    heard, _ = _hear(model, samples(pcm), language)
    return list(heard)


def transcribe(model: str, audio: bytes, language: str | None) -> tuple[str, float]:
    """Transcribe, returning the text AND the audio's duration in seconds.

    The duration is the billable quantity: transcription is priced per minute of
    audio by everyone who sells it, and the ai plane meters this call. It was
    already computed here and thrown away — `segments, _ = ...` — so audio billed
    zero no matter how much of it was sent. It is `info.duration`, the length of
    the audio SUBMITTED, not `duration_after_vad`, which is speech-only: a caller
    is charged for what they asked us to listen to, not for how much of it turned
    out to be talking.
    """
    heard, info = _hear(model, io.BytesIO(audio), language)
    text = " ".join(segment.text.strip() for segment in heard).strip()
    return text, info.duration

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


# How many decodes this process can really run at once.
#
# Measured on four idle cores, audio-seconds decoded per wall-second, medians of
# three interleaved rounds at 1, 2 and 4 concurrent sessions:
#
#   num_workers=1 (the default)   12.59   12.98   12.95
#   num_workers=2                 12.72   12.71   12.81
#   num_workers=4                 12.41   12.40   15.77
#
# Two things follow, and the first one is not what it looked like. The default
# ALREADY scales across concurrent sessions — it does not serialize them — so
# there was never a serialization defect here to fix, and num_workers=2 buys
# nothing measurable. Four is worth taking because it is the only setting that
# keeps full single-stream speed and also takes the best aggregate.
#
# cpu_threads is deliberately NOT set. Pinning it to a share of the pod is the
# obvious next step, and it costs a THIRD of single-stream speed (8.45 against
# 12.59) to buy 13-16% at concurrency. CTranslate2 sizes its own pool better than
# an arithmetic guess about how many cores there are.
#
# PARALLEL lives here rather than with the pool because it is a property of the
# MODEL, and the pool takes its size from here rather than naming its own.
PARALLEL = 4


@lru_cache(maxsize=None)
def load(model: str) -> WhisperModel:
    return WhisperModel(MODELS[model], device="cpu", compute_type="int8", num_workers=PARALLEL)


def samples(pcm: bytes) -> numpy.ndarray:
    """Raw little-endian int16 → the float32 in [-1, 1) the model consumes.

    Handing the model an array skips faster-whisper's demux entirely: a container
    goes through ffmpeg, an array does not. Raw audio has no container, so this is
    the shorter path AND the exact one.
    """
    return numpy.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0


def _hear(model: str, audio, language: str | None, words: bool = False):
    """The one call into the model. Both entries below differ only in what they
    hand it — a container to demux, or samples already in hand.

    `words` turns on per-word alignment. It is off by default because it is real
    extra work: the decoder produces segments on its way to the text, but word
    boundaries are a second pass over the cross-attention. Segment timings are
    therefore free and word timings are not, which is exactly why the caller
    says which it wants instead of us always paying for both.
    """
    return load(model).transcribe(audio, language=language, vad_filter=True, word_timestamps=words)


def segments(model: str, pcm: bytes, language: str | None) -> list:
    """Decode raw audio, returning the model's segments WITH their timings.

    A growing transcript needs the timings, not just the text: a segment that
    ended well before the audio does is settled and can be committed, while the
    tail is still moving. `transcribe` throws the timings away because a batch
    caller has nothing to do with them.
    """
    heard, _ = _hear(model, samples(pcm), language)
    return list(heard)


def transcribe(
    model: str, audio: bytes, language: str | None, words: bool = False
) -> tuple[str, float, list]:
    """Transcribe, returning the text, the audio's duration in seconds, and the
    segments it was heard as.

    The duration is the billable quantity: transcription is priced per minute of
    audio by everyone who sells it, and the ai plane meters this call. It was
    already computed here and thrown away — `segments, _ = ...` — so audio billed
    zero no matter how much of it was sent. It is `info.duration`, the length of
    the audio SUBMITTED, not `duration_after_vad`, which is speech-only: a caller
    is charged for what they asked us to listen to, not for how much of it turned
    out to be talking.

    The segments come back for the same reason the duration does: this function
    had them and dropped them. A caption cuts on a word boundary, and a caller
    who cannot see one has to guess where words fall by dividing the line's span
    by its letters — which drifts a little further with every line. Each segment
    carries its own words when `words` asked for them.
    """
    heard, info = _hear(model, io.BytesIO(audio), language, words)
    heard = list(heard)
    text = " ".join(segment.text.strip() for segment in heard).strip()
    return text, info.duration, heard

"""faster-whisper (CTranslate2, int8) — the STT half.

Weights bake into the image at build, so loading here is a read from the local
cache: lazy, once per process, kept for its life.
"""

import io
from functools import lru_cache

from faster_whisper import WhisperModel

# Request `model` → weight. `whisper` is the routed name the ai plane sends;
# distil covers English fast, `whisper-small` is the multilingual step up.
MODELS = {
    "whisper": "distil-small.en",
    "whisper-small": "small",
}


@lru_cache(maxsize=None)
def load(model: str) -> WhisperModel:
    return WhisperModel(MODELS[model], device="cpu", compute_type="int8")


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
    segments, info = load(model).transcribe(
        io.BytesIO(audio), language=language, vad_filter=True
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.duration

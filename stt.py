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


def transcribe(model: str, audio: bytes, language: str | None) -> str:
    segments, _ = load(model).transcribe(io.BytesIO(audio), language=language, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()

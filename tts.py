"""kokoro-onnx (82M) — the TTS half. wav natively; everything else via ffmpeg."""

import io
import subprocess
from functools import lru_cache

import soundfile
from kokoro_onnx import Kokoro
from pydantic import BaseModel

MODELS = {"kokoro": ("kokoro-v1.0.onnx", "voices-v1.0.bin")}
VOICES = ("af_heart", "af_bella", "am_michael")

# The OpenAI response formats, each mapped to the media type it REALLY is and the
# ffmpeg encode that produces it. wav needs no encode — soundfile writes it.
#
# This table replaced `if response_format == "wav" ... else mp3`, which answered
# every other format with an MP3 while the caller (and the gateway labelling the
# response) believed it was opus, aac or flac. Claiming a container we did not
# produce is the API lying to whatever has to play it, and a player handed the
# wrong container is entitled to refuse it. Every codec below is verified present
# in the image's ffmpeg (libopus, libmp3lame, aac, flac, pcm_s16le), so the
# honest answer here is to ENCODE what was asked for rather than to refuse it.
#
# opus is Ogg-contained, so it is audio/ogg — that is what the bytes are.
FORMATS: dict[str, tuple[str, list[str] | None]] = {
    "wav": ("audio/wav", None),
    "mp3": ("audio/mpeg", ["-f", "mp3", "-c:a", "libmp3lame"]),
    "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus"]),
    "aac": ("audio/aac", ["-f", "adts", "-c:a", "aac"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "pcm": ("audio/pcm", ["-f", "s16le", "-c:a", "pcm_s16le"]),
}


class Ask(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "mp3"


@lru_cache(maxsize=None)
def load(model: str) -> Kokoro:
    weights, voices = MODELS[model]
    return Kokoro(weights, voices)


def speak(ask: Ask) -> tuple[bytes, str]:
    """Synthesize, returning the audio and the media type it actually is.

    An unknown format raises: the caller validates against FORMATS first and
    answers 400, so this never silently substitutes one container for another.
    """
    if ask.response_format not in FORMATS:
        raise ValueError(f"unsupported response_format {ask.response_format!r}")

    samples, rate = load(ask.model).create(ask.input, voice=ask.voice)
    wav = io.BytesIO()
    soundfile.write(wav, samples, rate, format="WAV")

    mime, encode = FORMATS[ask.response_format]
    if encode is None:
        return wav.getvalue(), mime

    made = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *encode, "pipe:1"],
        input=wav.getvalue(),
        capture_output=True,
        check=True,
    )
    return made.stdout, mime

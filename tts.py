"""kokoro-onnx (82M) — the TTS half. wav natively; mp3 through ffmpeg."""

import io
import subprocess
from functools import lru_cache

import soundfile
from kokoro_onnx import Kokoro
from pydantic import BaseModel

MODELS = {"kokoro": ("kokoro-v1.0.onnx", "voices-v1.0.bin")}
VOICES = ("af_heart", "af_bella", "am_michael")


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
    samples, rate = load(ask.model).create(ask.input, voice=ask.voice)
    wav = io.BytesIO()
    soundfile.write(wav, samples, rate, format="WAV")
    if ask.response_format == "wav":
        return wav.getvalue(), "audio/wav"
    # OpenAI's default is mp3; ffmpeg is in the image for exactly this hop.
    made = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=wav.getvalue(),
        capture_output=True,
        check=True,
    )
    return made.stdout, "audio/mpeg"

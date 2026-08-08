"""The end-to-end test: real weights, real audio, a real decode.

Everything in test_speech.py replaces faster-whisper with a stand-in, because the
window arithmetic is what those tests are about and a stand-in makes it exact.
The cost of that is real: a suite where the decoder is only ever a stub cannot
tell a working decoder from a deleted one. This file closes that hole — nothing
here is stubbed, so it fails if the model is not loaded, if raw audio is handed
to it wrongly, or if the samples are scaled wrong.

The audio is genuine speech, spoken by this service's own other half: kokoro says
a sentence, ffmpeg renders it to exactly the shape a browser would send, and
whisper has to say it back. A round trip through both halves needs no fixture in
the tree and no recording of anyone's voice.

It runs where the weights are, which is the finished image — the fast suite runs
in a stage above the weight bake and never sees this file.
"""

import io
import time

import numpy
import pytest
import soundfile
from fastapi.testclient import TestClient

import main
import stt
import transcript
import tts

SAID = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs."
)
WORDS = ("quick", "brown", "fox", "lazy", "dog", "five", "dozen")

client = TestClient(main.app)


def speech(text: str) -> bytes:
    """Say it, then render it to what the wire carries: raw little-endian int16,
    mono, 16 kHz — no container, no header, exactly what a browser sends.

    Kokoro speaks at its own rate, so this resamples once, which is the same step
    the browser takes before it pushes.
    """
    samples, rate = tts.load("kokoro").create(text, voice="af_heart")
    want = int(len(samples) * stt.RATE / rate)
    at = numpy.linspace(0, len(samples) - 1, want)
    flat = numpy.interp(at, numpy.arange(len(samples)), samples)
    return (numpy.clip(flat, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def settle(live, timeout=180.0):
    """Wait for the background decode to stand down. A real decode on CPU takes
    seconds, which is exactly why the ack does not wait for one."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with live._lock:
            if not live._decoding:
                return
        time.sleep(0.02)
    raise AssertionError("decode never settled")


@pytest.fixture(scope="module")
def said() -> bytes:
    pcm = speech(SAID)
    assert len(pcm) > 4 * stt.SECOND, "the fixture must be several seconds of real speech"
    return pcm


def test_a_real_voice_becomes_a_real_transcript(said):
    """Push real speech through the real endpoint in real chunks and read it back.

    A decoder that is removed, never loaded, or handed the wrong sample scale
    returns nothing here — there is no stub to answer in its place.
    """
    live = client.post("/v1/audio/transcript", json={"model": "whisper", "language": "en"})
    assert live.status_code == 201, live.text
    tid = live.json()["id"]

    for at in range(0, len(said), transcript.CHUNK):
        r = client.post(f"/v1/audio/transcript/{tid}", content=said[at:at + transcript.CHUNK])
        assert r.status_code == 200, r.text

    shut = client.delete(f"/v1/audio/transcript/{tid}")
    assert shut.status_code == 200, shut.text

    heard = shut.json()["text"].lower()
    missing = [w for w in WORDS if w not in heard]
    assert not missing, f"whisper did not return {missing} from real speech; heard {heard!r}"


def test_it_decodes_while_the_transcript_is_still_open(said):
    """The point of the endpoint is words BEFORE the end. Push part of the audio,
    let the background decode finish, and read what it has — this is the live
    path, not the flush that close performs."""
    live = transcript.begin("whisper", "en")
    half = len(said) // 2 // stt.WIDTH * stt.WIDTH
    for at in range(0, half, transcript.CHUNK):
        live.push(said[at:at + transcript.CHUNK])

    settle(live)
    heard = (live.text + " " + live.pending).lower()
    assert any(w in heard for w in WORDS), f"nothing decoded mid-stream; heard {heard!r}"


def test_the_seconds_billed_are_the_seconds_spoken(said):
    """Metering, measured against audio whose length is known independently: the
    session must account for the whole utterance and no more."""
    live = transcript.begin("whisper", "en")
    billed = sum(
        live.push(said[at:at + transcript.CHUNK])
        for at in range(0, len(said), transcript.CHUNK)
    )
    spoken = len(said) / stt.SECOND
    assert billed == pytest.approx(spoken)
    assert live.seconds == pytest.approx(spoken)
    assert spoken > 4.0, "a real sentence, not a click"


def test_the_batch_route_still_reports_a_real_duration(said):
    """The quantity the plane already meters, from the real model rather than a
    stub: a wav of known length must come back with that length."""
    wav = io.BytesIO()
    soundfile.write(wav, numpy.frombuffer(said, dtype="<i2"), stt.RATE, format="WAV")
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav.getvalue(), "audio/wav")},
        data={"model": "whisper", "response_format": "verbose_json"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["duration"] == pytest.approx(len(said) / stt.SECOND, abs=0.1)
    assert "fox" in r.json()["text"].lower()

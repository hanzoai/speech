"""The end-to-end test: real weights, real audio, a real decode.

Everything in test_speech.py replaces faster-whisper with a stand-in, because the
window arithmetic is what those tests are about and a stand-in makes it exact.
The cost of that is real: a suite where the decoder is only ever a stub cannot
tell a working decoder from a deleted one. This file closes that hole — nothing
here is stubbed, so it fails if the model is not loaded, if raw audio is handed
to it wrongly, or if the samples are scaled wrong.

The audio is genuine speech, spoken by this service's own other half: kokoro says
a sentence, it is resampled to exactly the shape a browser would send, and
whisper has to say it back. A round trip through both halves needs no fixture in
the tree and no recording of anyone's voice.

It runs where the weights are, which is the finished image — the fast suite runs
in a stage above the weight bake and never sees this file.
"""

import io
import os
import subprocess
import sys
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


def test_the_process_exits_cleanly_mid_decode(said, tmp_path):
    """Shut down while a decode is running and the process must still exit 0.

    A thread sitting inside the model's C++ when the interpreter tears down makes
    the C++ runtime abort — `terminate called without an active exception`,
    SIGABRT. Every test still reports passed, because the failure is in the exit
    code and not in the report; in a pod it is a container that cannot drain on
    SIGTERM. Only real weights reproduce it: a stand-in has no native code to
    abort in.

    The model is loaded first and the child pauses after pushing, so shutdown
    lands mid-decode by construction rather than by luck.
    """
    raw = tmp_path / "said.pcm"
    raw.write_bytes(said)
    child = (
        "import time, stt, transcript;"
        "stt.load('whisper');"
        "live = transcript.begin('whisper', 'en');"
        f"pcm = open({str(raw)!r}, 'rb').read();"
        "[live.push(pcm[a:a + transcript.CHUNK])"
        " for a in range(0, len(pcm), transcript.CHUNK)];"
        "time.sleep(0.5)"
    )
    done = subprocess.run(
        [sys.executable, "-c", child],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, timeout=600,
    )
    assert done.returncode == 0, (
        f"exit {done.returncode}\n{done.stderr.decode()[-800:]}"
    )


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


# ── voice activity, against the real detector ───────────────────────────────
#
# test_speech.py replaces silero, because the squelch's arithmetic is what those
# tests are about. What it cannot tell is whether the real model agrees with the
# real room: whether a fan is silence and a word is not.

TURNS = ("Right, so the migration landed yesterday and the error rate is flat.",
         "Kubernetes rescheduled the workers twice overnight.")


def tone(seconds: float, dbfs: float = -45.0) -> bytes:
    """Room tone: a fan, a laptop, an open microphone in an empty room. Digital
    silence is a fixture no detector can fail, so this is not that — it is noise
    at a level a meeting really carries."""
    rng = numpy.random.default_rng(5)
    n = int(seconds * stt.RATE)
    warm = numpy.convolve(rng.standard_normal(n), numpy.ones(16) / 16, mode="same")
    return pcm(warm / (numpy.abs(warm).max() or 1.0) * 10 ** (dbfs / 20.0))


def pcm(x: numpy.ndarray) -> bytes:
    return (numpy.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def feed(live, audio: bytes) -> float:
    return sum(live.push(audio[at:at + transcript.CHUNK])
               for at in range(0, len(audio), transcript.CHUNK))


@pytest.fixture(scope="module")
def meeting() -> bytes:
    """A meeting: a lull, a turn, a pause inside it, another turn, a lull."""
    parts = [tone(2.5), speech(TURNS[0]), tone(0.5), speech(TURNS[1]), tone(2.5)]
    return b"".join(parts)


def test_an_empty_room_is_never_decoded(said):
    """The whole point, against the real detector. Six seconds of a room with
    nobody in it must cost nothing: no bytes to the decoder, no seconds to the
    bill. Then the same session hears a voice and pays for it — without which
    this test is also passed by a service that has stopped working."""
    live = transcript.begin("whisper", "en")

    assert feed(live, tone(6.0)) == 0.0, "a room with nobody in it is free"
    assert live.seconds == 0.0
    assert live._got == 0, "and reaches the decoder not at all"

    assert feed(live, said) > 0.0, "and a voice in the same room does not"


def test_a_push_carrying_no_audio_is_answered_not_refused():
    """The route takes whatever body arrives, and an empty one is whole int16
    frames and under the ceiling — so it reaches the squelch, and silero cannot
    be asked about nothing: it indexes the last window of what it is given, and
    of nothing there is none. A push with no audio was a no-op before there was
    anything in front of the decoder and it stays one, which is also how a client
    reads the newest text without sending more.

    It lives here rather than with the fast tests because only the real model
    has that edge — a stand-in answers an empty array quite happily, and a suite
    that only ever sees the stand-in passes either way."""
    live = client.post("/v1/audio/transcript", json={"model": "whisper"})
    tid = live.json()["id"]
    r = client.post(f"/v1/audio/transcript/{tid}", content=b"")
    assert r.status_code == 200, r.text
    assert r.json()["duration"] == 0.0


def test_the_word_after_a_pause_is_still_there(meeting):
    """The failure that would make this a bad trade: a squelch that opens on the
    trigger has already eaten the syllable that triggered it. Every word must
    survive a meeting made of turns, pauses and lulls — read back through the
    real decoder, which is the only judge of whether a syllable was lost."""
    live = transcript.begin("whisper", "en")
    billed = feed(live, meeting)
    live.close()

    heard = live.text.lower()
    said = " ".join(TURNS).lower()
    missing = [w.strip(".,") for w in said.split() if w.strip(".,") not in heard]
    assert not missing, f"the squelch lost {missing}; heard {heard!r}"

    sent = len(meeting) / stt.SECOND
    assert billed < sent, "and the lulls were not charged for"
    assert billed == pytest.approx(live.seconds)

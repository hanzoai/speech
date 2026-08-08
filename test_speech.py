"""The service's contract, tested without loading a single weight.

stt.transcribe and tts.speak are replaced per test: the models are baked into
the image and cost seconds to load, so exercising them here would test
faster-whisper and kokoro rather than this service. What IS this service's own
is the shape of what it returns — which is exactly where both defects lived.
"""

import time

import numpy
import pytest
from fastapi.testclient import TestClient

import main
import stt
import transcript
import tts

client = TestClient(main.app)


@pytest.fixture
def heard(monkeypatch):
    """Stub the transcriber; record the call, return a known text + duration."""
    calls = {}

    def fake(model, audio, language):
        calls.update(model=model, audio=audio, language=language)
        return "the quick brown fox", 12.5

    monkeypatch.setattr(stt, "transcribe", fake)
    return calls


@pytest.fixture
def spoken(monkeypatch):
    """Stub the synthesizer; echo back the format it was asked for."""
    def fake(ask):
        mime, _ = tts.FORMATS[ask.response_format]
        return b"AUDIOBYTES", mime

    monkeypatch.setattr(tts, "speak", fake)


# ── STT: the duration is the billable quantity ──────────────────────────────

def test_verbose_json_carries_duration(heard):
    """The defect: `segments, _ = ...transcribe(...)` discarded info.duration, so
    the ai plane metered every transcription at zero seconds no matter how much
    audio was sent. Transcription is billed per MINUTE, so a response with no
    duration cannot be billed at all."""
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
        data={"model": "whisper", "response_format": "verbose_json"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "the quick brown fox", "duration": 12.5}


def test_plain_json_is_unchanged(heard):
    """OpenAI's `json` body is {"text"}; a client reading the standard shape must
    not start receiving extra fields because we needed one internally."""
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"model": "whisper", "response_format": "json"},
    )
    assert r.json() == {"text": "the quick brown fox"}


def test_text_format_returns_plain_text(heard):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"model": "whisper", "response_format": "text"},
    )
    assert r.text == "the quick brown fox"
    assert r.headers["content-type"].startswith("text/plain")


def test_unknown_model_is_404(heard):
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"x", "audio/wav")},
        data={"model": "not-a-model"},
    )
    assert r.status_code == 404


# ── TTS: the container is what we say it is ─────────────────────────────────

@pytest.mark.parametrize(
    "fmt,mime",
    [
        ("wav", "audio/wav"),
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("aac", "audio/aac"),
        ("flac", "audio/flac"),
        ("pcm", "audio/pcm"),
    ],
)
def test_every_format_is_labelled_as_itself(spoken, fmt, mime):
    """The defect: `if response_format == "wav" ... else mp3` answered opus, aac,
    flac and pcm with an MP3, and the gateway labelled it from the REQUEST — so
    a caller asking for opus got `Content-Type: audio/opus` wrapping an MP3."""
    r = client.post("/v1/audio/speech", json={"model": "kokoro", "input": "hi", "response_format": fmt})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(mime)


def test_unsupported_format_is_refused_by_name(spoken):
    """Refuse, never substitute. The message names what IS available so a caller
    can act on it."""
    r = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "response_format": "midi"},
    )
    assert r.status_code == 400
    assert "midi" in r.text
    for fmt in tts.FORMATS:
        assert fmt in r.text


def test_default_format_is_mp3(spoken):
    """mp3 is OpenAI's default and the one every browser plays; hanzo.chat's
    playback path depends on it."""
    r = client.post("/v1/audio/speech", json={"model": "kokoro", "input": "hi"})
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_unknown_voice_is_400(spoken):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hi", "voice": "nobody"},
    )
    assert r.status_code == 400


def test_speak_refuses_unknown_format_even_when_called_directly():
    """The guard lives in tts.speak too, not only in the route: a second caller
    reaching the synthesizer directly cannot get a substituted container."""
    with pytest.raises(ValueError):
        tts.speak(tts.Ask(input="hi", response_format="midi"))


def test_formats_cover_the_openai_set():
    """Every format the OpenAI audio API defines is encodable in this image —
    verified against `ffmpeg -encoders` (libopus, libmp3lame, aac, flac,
    pcm_s16le). If one is dropped, callers silently lose it."""
    assert set(tts.FORMATS) == {"wav", "mp3", "opus", "aac", "flac", "pcm"}


# ── stt.transcribe itself: the duration must come from the model, not a stub ──

class _Segment:
    def __init__(self, text): self.text = text


class _Info:
    """faster-whisper's TranscriptionInfo, reduced to what billing reads."""
    duration = 7.25            # the audio SUBMITTED
    duration_after_vad = 3.10  # speech only — deliberately different


class _Model:
    def __init__(self): self.called_with = {}

    def transcribe(self, audio, language=None, vad_filter=None):
        self.called_with = dict(language=language, vad_filter=vad_filter)
        return iter([_Segment(" the quick "), _Segment(" brown fox ")]), _Info()


def test_transcribe_returns_the_models_duration(monkeypatch):
    """The defect lived HERE, not in the route: `segments, _ = ...` dropped the
    info that carries the duration. Stubbing the route's transcriber cannot see
    that, so this stubs the MODEL and exercises the real function — the test the
    absence of which let the first mutation of this fix pass unnoticed."""
    model = _Model()
    monkeypatch.setattr(stt, "load", lambda m: model)

    text, duration = stt.transcribe("whisper", b"RIFFxxxx", None)

    assert duration == 7.25, "the audio duration must reach the caller; 0 means unbillable"
    assert text == "the quick brown fox", "segments are joined and stripped"


def test_transcribe_bills_submitted_audio_not_speech_only(monkeypatch):
    """`duration` (submitted), never `duration_after_vad` (speech only). A caller
    is charged for what they asked us to listen to, not for how much of it turned
    out to be talking — and VAD is on, so the two always differ."""
    monkeypatch.setattr(stt, "load", lambda m: _Model())
    _, duration = stt.transcribe("whisper", b"x", None)
    assert duration == _Info.duration
    assert duration != _Info.duration_after_vad


def test_transcribe_forwards_the_language_hint(monkeypatch):
    model = _Model()
    monkeypatch.setattr(stt, "load", lambda m: model)
    stt.transcribe("whisper", b"x", "de")
    assert model.called_with["language"] == "de"
    assert model.called_with["vad_filter"] is True


# ── the growing transcript ──────────────────────────────────────────────────
#
# These stub the MODEL, never `stt.segments` or `Transcript` — so the real
# decode-window arithmetic runs. Stubbing the layer under test is how the first
# duration defect survived its own suite.
#
# The stand-in reads the audio it is HANDED and names each segment after the
# sample values inside it, so the text is a function of which bytes are in the
# window. A trim that drops the wrong bytes, drops too many, or drops none does
# not merely change a length here — it changes the words.


def tone(value: int, seconds: float) -> bytes:
    """PCM16 whose every sample is `value`: audio that says its own name."""
    return numpy.full(int(seconds * stt.RATE), value, dtype="<i2").tobytes()


class _Seg:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _Span:
    def __init__(self, span):
        self.duration = span
        self.duration_after_vad = span / 2


class _Ear:
    """faster-whisper's shape: cuts what it is given on a fixed grid and reads
    each piece's own samples back as its text."""

    STEP = 2.0

    def __init__(self):
        self.spans = []

    def transcribe(self, audio, language=None, vad_filter=None):
        span = len(audio) / stt.RATE
        self.spans.append(span)
        segs, at = [], 0.0
        while at < span - 1e-9:
            end = min(at + self.STEP, span)
            mid = int((at + end) / 2 * stt.RATE)
            segs.append(_Seg(str(round(float(audio[mid]) * 32768)), at, end))
            at = end
        return iter(segs), _Span(span)


@pytest.fixture
def ear(monkeypatch):
    """Put the stand-in under `stt.load`, so `stt.segments` and the whole window
    are the real code — only faster-whisper itself is replaced."""
    heard = _Ear()
    monkeypatch.setattr(stt, "load", lambda m: heard)
    transcript._live.clear()
    yield heard
    transcript._live.clear()


def settle(live, timeout=10.0):
    """Wait for the background decode to stand down — the ack deliberately does
    not, so a test that reads the text must."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with live._lock:
            if not live._decoding:
                return
        time.sleep(0.005)
    raise AssertionError("decode never settled")


def start(**over):
    body = {"model": "whisper", "language": "en"} | over
    r = client.post("/v1/audio/transcript", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ── metering: the reason this endpoint can exist at all ─────────────────────

def test_push_meters_the_audio_it_carried(ear):
    """`duration` is the seconds THIS push brought, named as the batch route
    names it so the plane meters both by one rule. A push that reports 0 is
    audio served free."""
    live = start()
    chunk = tone(7, 0.25)  # 8000 bytes
    r = client.post(f"/v1/audio/transcript/{live['id']}", content=chunk)
    assert r.status_code == 200, r.text
    assert r.json()["duration"] == 0.25
    assert r.json()["seconds"] == 0.25


def test_a_session_bills_every_second_once_and_only_once(ear):
    """The sum of what the pushes reported must equal the audio submitted —
    under-counting serves audio free, over-counting bills for silence that was
    never sent. Close adds no audio, so it must add no charge."""
    live = start()
    sent = 0.0
    billed = 0.0
    for n in range(12):  # 3 s in 250 ms pushes
        r = client.post(f"/v1/audio/transcript/{live['id']}", content=tone(n + 1, 0.25))
        billed += r.json()["duration"]
        sent += 0.25
    shut = client.delete(f"/v1/audio/transcript/{live['id']}")
    assert shut.status_code == 200, shut.text
    assert shut.json()["duration"] == 0.0, "close carries no audio, so it bills none"
    assert billed == pytest.approx(sent), "every second submitted, billed exactly once"
    assert shut.json()["seconds"] == pytest.approx(sent)


def test_seconds_come_from_the_bytes_not_from_the_decoder(ear):
    """Metering must not depend on the model returning anything. A decoder that
    hears silence still consumed the audio it was sent."""
    live = transcript.begin("whisper", None)
    assert live.push(tone(0, 1.5)) == 1.5
    assert live.seconds == 1.5


# ── the window: what is settled commits, what is still moving does not ──────

def test_settled_words_commit_and_their_audio_is_freed(ear):
    """A segment that ended a guard-length before the audio did cannot change, so
    it is committed and its bytes are dropped. The text names the samples, so a
    wrong trim shows up as wrong words."""
    live = transcript.begin("whisper", None)
    for value in (1, 2, 3):
        live.push(tone(value, 2.0))
    settle(live)

    assert live.text == "1 2", "the first two tones are settled"
    assert live.pending == "3", "the last is still open and stays revisable"
    assert len(live._window) == int(2.0 * stt.SECOND), "committed audio is freed"


def test_the_open_tail_is_never_committed_early(ear):
    """Everything within GUARD of the end stays pending: committing it would
    freeze a word that the next chunk may still change."""
    live = transcript.begin("whisper", None)
    live.push(tone(9, 1.5))
    settle(live)
    assert live.text == "", "nothing is settled yet"
    assert live.pending == "9"


def test_close_commits_the_remainder(ear):
    """Nothing follows a close, so the guard has nothing to protect — the tail is
    decoded and committed, or the last words are lost."""
    live = transcript.begin("whisper", None)
    for value in (1, 2, 3):
        live.push(tone(value, 2.0))
    settle(live)
    live.close()
    assert live.text == "1 2 3"
    assert live.pending == ""


def test_the_window_stays_bounded(ear):
    """Committed audio is dropped, so a long session decodes a window that stops
    growing instead of one that grows with the call."""
    live = transcript.begin("whisper", None)
    for value in range(1, 31):  # 60 s
        live.push(tone(value, 2.0))
        settle(live)
    assert len(live._window) <= int(transcript.WINDOW * stt.SECOND)
    assert max(ear.spans) <= transcript.WINDOW


def test_text_is_never_duplicated_across_trims(ear):
    """A commit that failed to free its audio would decode the same words again
    and say them twice. Each tone must appear exactly once."""
    live = transcript.begin("whisper", None)
    for value in range(1, 11):
        live.push(tone(value, 2.0))
        settle(live)
    live.close()
    assert live.text.split() == [str(v) for v in range(1, 11)]


# ── the surface ─────────────────────────────────────────────────────────────

def test_open_states_the_shape_it_wants(ear):
    live = start()
    assert live["rate"] == 16000 and live["channels"] == 1 and live["format"] == "pcm16"
    assert live["id"].startswith("atr_")
    assert live["chunk_ms"] == 256


def test_a_different_rate_is_refused_not_resampled(ear):
    """Raw audio carries no header, so the wrong rate is not an error downstream
    — it is gibberish that transcribes successfully."""
    r = client.post("/v1/audio/transcript", json={"model": "whisper", "rate": 8000})
    assert r.status_code == 400
    assert "16000" in r.text


def test_unknown_model_is_404_on_open(ear):
    r = client.post("/v1/audio/transcript", json={"model": "not-a-model"})
    assert r.status_code == 404


def test_unknown_transcript_is_404(ear):
    r = client.post("/v1/audio/transcript/atr_nope", content=tone(1, 0.25))
    assert r.status_code == 404


def test_oversize_chunk_is_refused_not_truncated(ear):
    live = start()
    r = client.post(f"/v1/audio/transcript/{live['id']}", content=tone(1, 3.0))
    assert r.status_code == 413


def test_half_a_sample_is_refused(ear):
    """An odd byte count is not int16 frames; accepting it shifts every sample
    after it by one byte and turns the rest of the stream into noise."""
    live = start()
    r = client.post(f"/v1/audio/transcript/{live['id']}", content=b"\x01\x02\x03")
    assert r.status_code == 400


def test_a_full_transcript_stops_accepting_audio(ear):
    live = transcript.begin("whisper", None)
    live.seconds = transcript.LIMIT
    r = client.post(f"/v1/audio/transcript/{live.id}", content=tone(1, 0.25))
    assert r.status_code == 409


def test_closing_twice_is_404(ear):
    live = start()
    assert client.delete(f"/v1/audio/transcript/{live['id']}").status_code == 200
    assert client.delete(f"/v1/audio/transcript/{live['id']}").status_code == 404


def test_abandoned_transcripts_are_collected(ear):
    """A client that stops talking must not pin its audio in memory forever."""
    live = transcript.begin("whisper", None)
    live.push(tone(1, 0.5))
    live._touched -= transcript.IDLE + 1
    transcript.begin("whisper", None)  # collection happens on the way in
    assert transcript.find(live.id) is None


def test_samples_map_int16_onto_the_unit_range(ear):
    """The model wants float in [-1, 1). Getting this wrong does not fail — it
    transcribes quiet audio as silence and loud audio as noise."""
    got = stt.samples(numpy.array([0, 16384, -32768], dtype="<i2").tobytes())
    assert got.dtype == numpy.float32
    assert list(got) == [0.0, 0.5, -1.0]


class _Deaf:
    """A model that fails, to check the failure is visible rather than quiet."""

    def transcribe(self, audio, language=None, vad_filter=None):
        raise RuntimeError("decoder is unhappy")


def test_a_failed_decode_says_so(monkeypatch, caplog):
    """A decode runs in a pool whose Future is discarded, so an exception in it
    reaches nobody: pushes keep acking 200 with the right `duration` and the
    transcript is simply always empty. Failure has to be loud enough to find."""
    monkeypatch.setattr(stt, "load", lambda m: _Deaf())
    transcript._live.clear()
    live = transcript.begin("whisper", None)
    with caplog.at_level("ERROR"):
        live.push(tone(1, 2.0))
        settle(live)
    assert "decoder is unhappy" in caplog.text
    assert live.text == "" and live.pending == ""

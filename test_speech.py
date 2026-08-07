"""The service's contract, tested without loading a single weight.

stt.transcribe and tts.speak are replaced per test: the models are baked into
the image and cost seconds to load, so exercising them here would test
faster-whisper and kokoro rather than this service. What IS this service's own
is the shape of what it returns — which is exactly where both defects lived.
"""

import pytest
from fastapi.testclient import TestClient

import main
import stt
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

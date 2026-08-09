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

import bench
import main
import stt
import transcript
import tts
import vad

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


# ── voice activity: the decode that never runs ──────────────────────────────
#
# Silero is a weight like every other, so it is replaced here and what runs is
# the squelch's own arithmetic — the two thresholds, the hangover, the lead-in.
# Everything above this line is about the decode window, so everything above
# this line gets a room that hears a voice in all of it and is untouched.


@pytest.fixture(autouse=True)
def talking(monkeypatch):
    monkeypatch.setattr(
        vad, "get_vad_model",
        lambda: lambda heard: numpy.ones((len(heard) // vad.STEP, 1)),
    )


class _Room:
    """silero's shape: one probability per window, read from how loud that window
    is — so a test writes speech and silence as loud and quiet audio, and the
    machine that decides between them is the real one."""

    def __call__(self, heard):
        return numpy.abs(heard.reshape(-1, vad.STEP)).max(axis=1).reshape(-1, 1)


@pytest.fixture
def room(monkeypatch):
    monkeypatch.setattr(vad, "get_vad_model", _Room)


def loud(seconds: float) -> bytes:
    return tone(30000, seconds)  # 0.92 of full scale


def quiet(seconds: float) -> bytes:
    return tone(3, seconds)  # 0.0001 — a room with nobody in it


def test_a_quiet_room_costs_nothing(ear, room):
    """The whole point. Silence decodes to "" for the same CPU and the same money
    as speech, so an attendant left in an empty room bills a tenant continuously
    for nothing. Dropped audio must reach neither the decoder nor the meter."""
    live = transcript.begin("whisper", None)
    billed = sum(live.push(quiet(0.25)) for _ in range(12))  # 3 s of nobody

    assert billed == 0.0, "silence must not be billed"
    assert live.seconds == 0.0
    assert ear.spans == [], "and must not reach the decoder at all"
    assert len(live._window) == 0


def test_a_voice_is_decoded(ear, room):
    """The other direction, which is what makes the test above mean anything: a
    squelch that dropped everything would satisfy that one perfectly."""
    live = transcript.begin("whisper", None)
    billed = sum(live.push(loud(0.25)) for _ in range(8))
    settle(live)

    assert billed == pytest.approx(2.0)
    assert ear.spans, "speech must reach the decoder"


def test_the_onset_that_opened_the_squelch_is_decoded_with_it(ear, room):
    """The classic failure, and the one that costs more than it saves: a voice is
    not detected until it is already under way, so a squelch that begins at the
    trigger has already eaten the first syllable. The audio from before the
    trigger is kept, and handed over by the push that opens."""
    live = transcript.begin("whisper", None)
    for _ in range(8):
        assert live.push(quiet(0.25)) == 0.0

    opened = live.push(loud(0.25))

    assert opened > 0.25, "the push that opens carries more than itself"
    assert bytes(live._window[-len(loud(0.25)):]) == loud(0.25)
    before = bytes(live._window[:-len(loud(0.25))])
    assert before == quiet(len(before) / stt.SECOND), "and what precedes it is what preceded"
    assert len(before) == int(vad.LEAD * stt.SECOND), "as much of it as LEAD says"
    assert opened == pytest.approx(len(live._window) / stt.SECOND), "all of it billed once"


def test_a_pause_inside_a_sentence_does_not_end_it(ear, room):
    """Speech is not continuous: there is a gap between words and a breath
    between clauses. Closing on the first quiet window would chop a sentence into
    fragments and clip the word after every pause."""
    live = transcript.begin("whisper", None)
    live.push(loud(0.25))
    gap = [live.push(quiet(0.25)) for _ in range(2)]  # half a second
    after = live.push(loud(0.25))

    assert gap == [0.25, 0.25], "a pause this short is inside the sentence"
    assert after == 0.25, "so the word after it needs no lead-in — it never closed"


def test_the_squelch_closes_once_the_room_stays_quiet(ear, room):
    """And it must actually close, or the saving is only ever deferred. The
    hangover is spent — HANG seconds of quiet decode — and then nothing does."""
    live = transcript.begin("whisper", None)
    live.push(loud(0.25))
    after = [live.push(quiet(0.25)) for _ in range(8)]  # 2 s of nobody

    spent = sum(after)
    assert spent == pytest.approx(0.75), "the hangover, and no more than it"
    assert spent <= vad.HANG
    assert after[3:] == [0.0] * 5, "quiet from there on is free"


def test_a_session_never_bills_a_second_it_did_not_decode(ear, room):
    """The invariant that makes the lead-in safe to charge for: the push that
    opens reports more than its own length, so the arithmetic has to hold over
    the session rather than over one call. Every second billed is a second a
    decoder read, and no second is billed twice."""
    live = transcript.begin("whisper", None)
    sent = billed = 0.0
    for piece in (quiet(1.0), loud(1.0), quiet(1.0), loud(1.0), quiet(2.0)):
        for at in range(0, len(piece), transcript.CHUNK):
            billed += live.push(piece[at:at + transcript.CHUNK])
            sent += len(piece[at:at + transcript.CHUNK]) / stt.SECOND

    assert billed == pytest.approx(live.seconds), "the acks and the session agree"
    assert live.seconds == pytest.approx(live._got / stt.SECOND), "billed is decoded"
    assert billed < sent, "and the quiet room was not charged for"


class _Level:
    """A room at one stated probability — it reads the level, not the audio — so
    a test can write the window silero is least sure about."""

    def __init__(self):
        self.p = 0.0

    def __call__(self, heard):
        return numpy.full((len(heard) // vad.STEP, 1), self.p)


def test_a_window_the_model_is_unsure_of_holds_its_ground(monkeypatch):
    """Between the two thresholds silero is not sure, and one threshold cannot
    express that: the same probability must continue speech without being able to
    begin it. With one threshold, a syllable fading through it ends the sentence
    and a room's own hum starts one."""
    level = _Level()
    monkeypatch.setattr(vad, "get_vad_model", lambda: level)
    squelch = vad.Squelch()

    level.p = (vad.OPEN + vad.SHUT) / 2
    assert squelch.admit(tone(1, 0.25)) == b"", "unsure does not begin speech"

    level.p = vad.OPEN
    assert squelch.admit(tone(1, 0.25)), "certain does"

    level.p = (vad.OPEN + vad.SHUT) / 2
    assert squelch.admit(tone(1, 2.0)), "and unsure does not end it, however long"


def test_the_ack_reports_zero_for_silence(ear, room):
    """`duration` on the ack is the field the ai plane meters. Whatever the
    session believes, this is the number that becomes money."""
    live = start()
    r = client.post(f"/v1/audio/transcript/{live['id']}", content=quiet(0.25))
    assert r.status_code == 200, r.text
    assert r.json()["duration"] == 0.0
    assert r.json()["seconds"] == 0.0


# ── which process owns the session ──────────────────────────────────────────

def test_open_names_where_it_lives(ear, monkeypatch):
    """A growing transcript lives in ONE process's memory, and a Service address
    round-robins per connection — so with two replicas half the pushes land on a
    replica that has never heard of the session and answer 404. `open` says which
    pod took it, and the caller addresses that pod for the rest of the session.

    Both directions are asserted: set, it is a real address; unset, it is empty
    rather than a guess, which is the honest answer for a single process."""
    import importlib

    monkeypatch.setenv("POD_IP", "10.244.3.17")
    monkeypatch.setenv("PORT", "8000")
    here = importlib.reload(main)
    assert here.HERE == "http://10.244.3.17:8000"

    monkeypatch.delenv("POD_IP")
    nowhere = importlib.reload(main)
    assert nowhere.HERE == ""
    # And the field is on the open response, not merely computed.
    assert "at" in nowhere.transcript_open(nowhere.Open(model="whisper"))


# ── fairness: every session gets a turn ─────────────────────────────────────

def test_no_session_is_starved_by_another(monkeypatch):
    """A worker used to loop until its OWN window stopped growing, which under
    continuous audio is never — new bytes always arrive mid-decode. The pool's
    slots were then held by whichever sessions reached them first, for as long as
    those sessions kept receiving audio, which in a meeting is the whole meeting.

    Measured on the real decoder with three concurrent sessions, one ran ZERO
    passes across its entire 70 s: it returned empty text and was billed for every
    second of it, and its window grew to 69.9 s because the WINDOW cap lives in
    the pass that never ran.

    The decode here is SLOW on purpose. With an instant one the pool is never
    contended and every policy looks fair.
    """
    import collections
    import threading

    seen = collections.Counter()
    guard = threading.Lock()

    def slow(model, pcm, language):
        with guard:
            seen[pcm[:1]] += 1  # each session pushes its own marker byte
        time.sleep(0.06)
        return []

    monkeypatch.setattr(stt, "segments", slow)

    sessions = [transcript.begin("whisper", None) for _ in range(3)]
    marks = [bytes([7 + i]) for i in range(len(sessions))]
    chunk = int(FLOOR_BYTES := 1.0 * stt.SECOND)  # one FLOOR of audio per push

    stop = time.monotonic() + 2.0

    def feed(live, mark):
        while time.monotonic() < stop:
            live.push(mark * chunk)
            time.sleep(0.02)

    threads = [threading.Thread(target=feed, args=(s, m)) for s, m in zip(sessions, marks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Asserted while the audio is still arriving, not after everything drains:
    # the question is whether a session gets served DURING a busy call, not
    # whether it is served eventually once the others stop.
    starved = [i for i, m in enumerate(marks) if seen[m] == 0]
    assert not starved, f"sessions {starved} ran zero decode passes while the others ran {dict(seen)}"

    # Quiesce before the patch is undone. A pass still queued when `slow` is
    # removed would run against the REAL decoder and pull weights off the network
    # in the middle of the suite.
    for live in sessions:
        with live._lock:
            live._closed = True
        transcript.drop(live.id)
    end = time.monotonic() + 5.0
    for live in sessions:
        while time.monotonic() < end:
            with live._lock:
                if not live._decoding:
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("a decode never stood down")


# ── the capacity harness (bench.py) ─────────────────────────────────────────
#
# The harness answers "how far behind is the transcript", and a wrong answer
# there is worse than no answer: it is a capacity number someone provisions
# against. These check it against cases whose answer is known by construction —
# a session keeping up, and a session that has stopped decoding entirely.

def test_the_word_map_lands_on_the_seconds_it_was_built_from():
    """Recognized words become a position in the audio, exactly at the ends and
    exactly on repeat: the corpus is said once and tiled, so word N + all-words
    must land one whole corpus later."""
    sheet = bench.Sheet([(4, 2.0), (10, 5.0)], 5.0)
    assert sheet.at(0) == 0.0
    assert sheet.at(4) == 2.0
    assert sheet.at(10) == 5.0
    assert sheet.at(14) == 7.0  # one lap and four words
    assert sheet.at(7) == pytest.approx(3.5)  # halfway through the second sentence


def test_the_verdict_is_the_slope_and_a_stall_reads_as_one():
    """The number a capacity claim rests on. A transcript that keeps up adds no
    lag; one that has stopped adds a second of lag per second, which is 60 per
    minute — and the flush at close must not be allowed to flatter it."""
    keeping = [{"t": at, "heard_s": at - 2} for at in range(0, 600, 5)]
    stalled = [{"t": at, "heard_s": 40.0} for at in range(0, 600, 5)]
    assert bench.slope(keeping, 300) == pytest.approx(0.0, abs=0.01)
    assert bench.slope(stalled, 300) == pytest.approx(60.0, abs=0.01)
    assert bench.slope(stalled + [{"t": 601, "heard_s": 600, "shut": True}], 300) \
        == pytest.approx(60.0, abs=0.01)


def test_half_speed_reads_as_half_speed():
    """Between the two ends the reading has to be proportional, or the knee it
    reports is at the wrong concurrency."""
    half = [{"t": at, "heard_s": at / 2} for at in range(0, 600, 5)]
    assert bench.slope(half, 300) == pytest.approx(30.0, abs=0.01)

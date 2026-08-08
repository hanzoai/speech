"""A transcript that grows: audio arrives in pieces, text comes back on the ack.

Whisper is not a streaming model — it decodes a span of audio, not a sample at a
time. So a growing transcript is a window that is re-decoded as it fills, and the
art is deciding which of the words it returns are settled. A segment that ended
well before the audio did will not change when more audio arrives, so it is
committed and its audio is dropped; the tail stays open and is decoded again next
time. That is what keeps the window bounded and the committed text stable.

The ack never waits for a decode. A push appends its audio and returns the newest
text the session has, so the answer costs one memcpy and the round trip — decode
latency lands on a later ack instead of on this one. One decode runs per session
at a time, which is also what stops a fast pusher from queueing work faster than
the CPU retires it.

`seconds` is arithmetic on the bytes received, so it is the audio submitted,
exactly, and it cannot silently become zero: it is the same quantity `transcribe`
reports as `duration`, measured the same way, and it is what the call is billed on.
"""

import secrets
import threading
import time

import stt

# 250 ms per push at 16 kHz mono int16. Small enough to sit inside one ZAP frame,
# which the transport requires: a request is built as a single frame, so audio
# that does not fit in one cannot be sent at all.
CHUNK = 8 * 1024
CEILING = 64 * 1024  # 2 s; a larger push is refused rather than truncated

FLOOR = 1.0  # audio below this is not worth a decode pass
GUARD = 1.0  # trailing seconds never committed: the last word may still be moving
WINDOW = 12.0  # past this the guard is dropped, so the window cannot grow forever
IDLE = 30.0  # a session untouched this long is collectable
LIMIT = 600.0  # audio one session will accept, in seconds


class Transcript:
    """One growing transcript. Everything mutable is under `_lock`."""

    def __init__(self, model: str, language: str | None):
        self.id = "atr_" + secrets.token_hex(12)
        self.model = model
        self.language = language
        self.text = ""  # committed: settled, never revised
        self.pending = ""  # the open tail: newest decode, may still change
        self.seconds = 0.0  # audio received — the billable quantity
        self._window = bytearray()
        self._got = 0  # bytes ever received; marks progress for the worker
        self._lock = threading.Lock()
        self._decoding = False
        self._closed = False
        self._touched = time.monotonic()

    @property
    def full(self) -> bool:
        return self.seconds >= LIMIT

    def stale(self, now: float) -> bool:
        return now - self._touched > IDLE

    def push(self, pcm: bytes) -> float:
        """Accept audio; return the seconds THIS push carried.

        The return value is the metered quantity for this call — per push, not
        cumulative, so summing the pushes of a session yields the audio submitted
        once and only once.
        """
        with self._lock:
            self._window += pcm
            self._got += len(pcm)
            self.seconds += len(pcm) / stt.SECOND
            self._touched = time.monotonic()
            start = not self._decoding and len(self._window) >= FLOOR * stt.SECOND
            if start:
                self._decoding = True
        if start:
            threading.Thread(target=self._work, daemon=True).start()
        return len(pcm) / stt.SECOND

    def close(self) -> None:
        """Decode what is left and commit all of it — nothing follows, so nothing
        is held back. Blocks: the final text is the point of closing."""
        with self._lock:
            self._closed = True  # stops an in-flight decode from committing twice
            rest, self._window = bytes(self._window), bytearray()
        if not rest:
            return
        tail = " ".join(s.text.strip() for s in stt.segments(self.model, rest, self.language))
        with self._lock:
            self.text = (self.text + " " + tail).strip()
            self.pending = ""

    def _work(self) -> None:
        """Decode the window until it stops growing, then stand down.

        `_decoding` is cleared under the same lock that reads progress, so a push
        that arrives at the moment the worker gives up still starts a new one.
        """
        try:
            while True:
                with self._lock:
                    window, mark = bytes(self._window), self._got
                    if self._closed or len(window) < FLOOR * stt.SECOND:
                        self._decoding = False
                        return
                self._absorb(window)
                with self._lock:
                    if self._got == mark:
                        self._decoding = False
                        return
        except BaseException:
            with self._lock:
                self._decoding = False
            raise

    def _absorb(self, window: bytes) -> None:
        heard = stt.segments(self.model, window, self.language)
        span = len(window) / stt.SECOND
        # Past WINDOW the guard is dropped: holding the tail back forever would let
        # the window — and every decode over it — grow without bound.
        edge = span if span > WINDOW else span - GUARD

        done, cut = [], 0.0
        for s in heard:
            if s.end > edge:
                break  # segments are in time order: once one is open, the rest are
            done.append(s.text.strip())
            cut = s.end
        open_tail = " ".join(s.text.strip() for s in heard if s.end > cut)

        with self._lock:
            if self._closed:
                return
            if done:
                self.text = (self.text + " " + " ".join(done)).strip()
                del self._window[: int(cut * stt.RATE) * stt.WIDTH]
            self.pending = open_tail

    def state(self, billed: float = 0.0) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "text": self.text,
                "pending": self.pending,
                "seconds": round(self.seconds, 3),
                # What THIS call consumed, named as `transcribe` names it, so the
                # plane meters a push and a batch transcription by one rule.
                "duration": round(billed, 3),
            }


_live: dict[str, Transcript] = {}
_guard = threading.Lock()


def begin(model: str, language: str | None) -> Transcript:
    """Open a transcript, collecting abandoned ones on the way in.

    Sweeping here rather than on a timer means there is no clock to run and no
    task to supervise: sessions are only worth collecting when new ones arrive.
    """
    live = Transcript(model, language)
    now = time.monotonic()
    with _guard:
        for tid in [tid for tid, t in _live.items() if t.stale(now)]:
            del _live[tid]
        _live[live.id] = live
    return live


def find(tid: str) -> Transcript | None:
    with _guard:
        return _live.get(tid)


def drop(tid: str) -> Transcript | None:
    with _guard:
        return _live.pop(tid, None)

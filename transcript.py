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

Audio reaches the window through a squelch, so a quiet room reaches nothing at
all: no bytes, no decode, no charge. `seconds` is arithmetic on the bytes that
got through, and those are exactly the bytes a decoder read — so the bill is the
audio we listened to. It is the same quantity `transcribe` reports as `duration`,
measured the same way, it can never exceed the audio submitted, and it cannot
silently become zero for audio that WAS decoded, because it is the length of the
decoded bytes and nothing else.
"""

import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import stt
import vad

log = logging.getLogger(__name__)

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

# Decoding is CPU-bound, and the pool is exactly as wide as the model can really
# serve (stt.PARALLEL). Wider is not more throughput — CTranslate2 serializes
# calls on one model — it is only a longer queue in front of the same worker.
#
# A pool, not a thread per session, for a second reason that is not about speed:
# its workers are joined when the interpreter shuts down. Raw daemon threads are
# not, and a thread still inside the model's C++ when the runtime tears down
# aborts the process — `terminate called without an active exception`, SIGABRT,
# with every test passing and the run still red. In a pod that is a container
# that cannot drain on SIGTERM.
#
# THE QUEUE IS THE FAIRNESS. A pass is submitted, runs, and stands down, so the
# order sessions are served in is the pool's FIFO order. A worker that looped
# until its own window stopped growing never stood down under continuous audio,
# and the slots were held by whichever sessions reached them first.
_pool = ThreadPoolExecutor(max_workers=stt.PARALLEL, thread_name_prefix="decode")


class Transcript:
    """One growing transcript. Everything mutable is under `_lock`."""

    def __init__(self, model: str, language: str | None):
        self.id = "atr_" + secrets.token_hex(12)
        self.model = model
        self.language = language
        self.text = ""  # committed: settled, never revised
        self.pending = ""  # the open tail: newest decode, may still change
        self.seconds = 0.0  # audio decoded — the billable quantity
        self._window = bytearray()
        self._got = 0  # bytes ever received; marks progress for the worker
        self._voice = vad.Squelch()
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
        """Accept audio; return the seconds THIS push put in front of a decoder.

        That is the metered quantity for this call — per push, not cumulative, so
        summing the pushes of a session yields the audio decoded once and only
        once. A push the squelch drops returns 0.0: it was never decoded, so
        there is nothing to charge for. A push that OPENS the squelch carries the
        lead-in with it and so reports more than its own length; those seconds
        were withheld from every earlier push, so the session still bills each
        second at most once and never bills one it did not decode.

        The squelch runs under the lock because its memory of the room is part of
        the session's state, and it is a millisecond of arithmetic — no decode
        ever holds this lock.
        """
        with self._lock:
            self._touched = time.monotonic()
            heard = self._voice.admit(pcm)
            if not heard:
                return 0.0
            self._window += heard
            self._got += len(heard)
            self.seconds += len(heard) / stt.SECOND
            start = not self._decoding and len(self._window) >= FLOOR * stt.SECOND
            if start:
                self._decoding = True
        if start:
            _pool.submit(self._work)
        return len(heard) / stt.SECOND

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
        """Decode the window ONCE, then hand the worker back.

        It used to loop until the window stopped growing, and under continuous
        audio that is never: new bytes always arrive mid-decode, so a worker that
        started never returned. The two pool slots were then held by whichever
        sessions reached them first, for as long as those sessions kept receiving
        audio — which in a meeting is the whole meeting. Measured on three
        concurrent sessions, one ran ZERO passes across its entire 70 s, returned
        empty text, and was billed for every second of it. Its window was never
        trimmed either, because the WINDOW cap lives in the pass that never ran:
        69.9 s retained where the cap says 12.

        Standing down after one pass makes the pool's FIFO queue decide the order,
        so every session gets a turn.
        """
        try:
            with self._lock:
                window, mark = bytes(self._window), self._got
                if self._closed or len(window) < FLOOR * stt.SECOND:
                    self._decoding = False
                    return
            self._absorb(window)
        except BaseException:
            # The pool keeps a worker's exception inside its Future, and this one
            # is discarded — so without this the transcript simply stays empty
            # while every push still acks 200. Logged BEFORE standing down, so
            # anything watching the flag sees the reason first. The next push
            # starts a fresh attempt.
            log.exception("decode failed for %s", self.id)
            with self._lock:
                self._decoding = False
            raise
        # Audio that arrived DURING the pass is decoded on the next one, submitted
        # from the BACK of the queue. The stand-down condition is unchanged — no
        # new bytes since `mark` means there is nothing to decode again — so what
        # differs is only that the worker is handed back between passes instead of
        # held. `_decoding` stays true across the hand-off, so a push arriving
        # right now does not start a second worker for this transcript: still one
        # decode per session at a time.
        with self._lock:
            self._decoding = not self._closed and self._got != mark
            again = self._decoding
        if again:
            _pool.submit(self._work)

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

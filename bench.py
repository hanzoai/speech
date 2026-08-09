"""What this service sustains, measured — one tool, three verbs.

    say    speak a paragraph into the wire format, with a word-to-time map
    cost   what ONE decode pass costs, against window length and concurrency
    talk   drive N concurrent sessions in real time and watch the lag

Why lag and not latency. A push submits its audio and returns whatever text the
session happens to have, so the ack costs one memcpy whether the decoder is idle
or minutes behind: measured, it sat at 3.8 ms while the backlog grew fourfold.
Ack latency cannot see saturation. What moves is BACKLOG — audio that arrived and
has not become words — so `talk` measures position instead of latency. The corpus
is speech whose word count at every instant is known, so a recognized word count
maps back to a second of audio, and

    lag = wall seconds elapsed - seconds of audio already transcribed

is what a person in the meeting feels. It has two halves and both are reported,
because they fail differently: SUBMIT lag is audio the microphone has produced and
not yet sent (the transport is half duplex, so a slow ack holds the microphone
back), DECODE lag is audio the service holds and has not decoded.

A concurrency is sustainable when lag stops growing, so the verdict is the SLOPE
over the second half of the run, in seconds of lag added per minute of meeting.
Terminal lag cannot say this on its own — a diverging session looks fine early,
which is how seconds-long runs came to predict eight concurrent meetings and were
wrong by an order of magnitude. Runs must be minutes long: the stall that decides
the answer begins around four minutes in.

`cost` is the floor under any capacity claim. Whisper always processes a thirty
second mel window, so a pass over one second of audio costs nearly what a pass
over eight costs, and capacity is a budget of PASSES, not of audio. Past thirty
seconds of window a pass costs several times more, which is the cliff a session
falls off when its window grows faster than the decoder empties it.

Nothing here is stubbed and nothing in the service is instrumented for it: every
number comes from the ack the real endpoint already returns.
"""

import argparse
import http.client
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import stt

WORD = re.compile(r"[a-z0-9']+")

PARA = [
    "Good morning everyone, thanks for making the time today.",
    "I want to start with where the numbers landed last quarter.",
    "Revenue came in ahead of plan, but the mix shifted quite a bit.",
    "Most of the growth was in the smaller accounts rather than the large ones.",
    "That changes how we should think about the sales team next year.",
    "The second thing is the migration, which is roughly two thirds done.",
    "We hit an unexpected wall with the older storage layer.",
    "It turns out the schema had drifted in three separate places.",
    "None of that was visible until we tried to read the archive.",
    "So the estimate slipped by about two weeks, which I think is fine.",
    "On hiring, we made two offers and both of them were accepted.",
    "They start at the beginning of next month, both on the platform side.",
    "I would like each of you to think about who they pair with.",
    "Now, the part I actually want to argue about is pricing.",
    "The current model rewards exactly the customers we do not want.",
    "Somebody who sends us a huge burst pays almost nothing extra.",
    "Meanwhile the steady, predictable accounts subsidize all of it.",
    "I do not think a small tweak fixes that, it needs a rethink.",
    "My proposal is that we meter the resource rather than the request.",
    "That is harder to explain, but it is honest, and it scales.",
    "Before we go further, does anyone want to push back on any of this?",
    "Alright, let me hand over and we can walk through the details.",
    "One last thing, the review is moved to Thursday afternoon.",
    "Please read the document before then rather than during the meeting.",
]


# ── say ─────────────────────────────────────────────────────────────────────

def say(voice: str, out: str) -> None:
    """Kokoro speaks the paragraph, one sentence at a time, so every sentence's
    duration is known — that is what makes a word count convertible to a second.
    Said once: `talk` tiles it, so a session of any length costs one generation
    and the map stays exact by periodicity."""
    import numpy

    import tts

    said, marks, count = bytearray(), [], 0
    for line in PARA:
        samples, rate = tts.load("kokoro").create(line, voice=voice)
        want = int(len(samples) * stt.RATE / rate)
        at = numpy.linspace(0, len(samples) - 1, want)
        flat = numpy.interp(at, numpy.arange(len(samples)), samples)
        said += (numpy.clip(flat, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        count += len(line.split())
        marks.append((count, round(len(said) / stt.SECOND, 3)))
    open(out + ".pcm", "wb").write(bytes(said))
    json.dump({"seconds": len(said) / stt.SECOND, "voice": voice, "marks": marks},
              open(out + ".json", "w"))
    print(f"{out}: {len(said)/stt.SECOND:.1f}s, {count} words")


# ── cost ────────────────────────────────────────────────────────────────────

def burned() -> float:
    """CPU seconds this process has used, user + system, all threads."""
    at = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    return (int(at[11]) + int(at[12])) / 100.0


def cost(corpus: str, reps: int, threads: int, spans: list) -> None:
    said = open(corpus + ".pcm", "rb").read()
    stt.load("whisper")  # in the pod this happens once, at boot
    for span in spans:
        need = int(span * stt.SECOND)
        pool = said * (need // len(said) + 2)  # tile when a span outruns the corpus
        cuts = [pool[(i * need) % (len(pool) - need):][:need] for i in range(reps * threads)]

        def one(pcm: bytes) -> float:
            at = time.monotonic()
            stt.segments("whisper", pcm, "en")
            return time.monotonic() - at

        was, wall = burned(), time.monotonic()
        with ThreadPoolExecutor(max_workers=threads) as run:
            took = list(run.map(one, cuts))
        wall, cpu = time.monotonic() - wall, burned() - was
        print(json.dumps({
            "span_s": span,
            "threads": threads,
            "pass_median_s": round(statistics.median(took), 3),
            "pass_rtf": round(statistics.median(took) / span, 3),
            # Audio seconds retired per wall second, all decoders together.
            "throughput_x": round(len(took) * span / wall, 3),
            "cpu_s_per_pass": round(cpu / len(took), 3),
        }), flush=True)


# ── talk ────────────────────────────────────────────────────────────────────

class Sheet:
    """Recognized words -> seconds of audio, for a corpus tiled end to end."""

    def __init__(self, marks: list, seconds: float):
        self.marks = [(int(w), float(s)) for w, s in marks]
        self.words, self.seconds = self.marks[-1][0], seconds

    @classmethod
    def read(cls, corpus: str) -> "Sheet":
        raw = json.load(open(corpus + ".json"))
        return cls(raw["marks"], raw["seconds"])

    def at(self, words: int) -> float:
        loops, rest = divmod(words, self.words)
        base, was_w, was_s = loops * self.seconds, 0, 0.0
        for w, s in self.marks:
            if rest <= w:
                span = w - was_w
                return base + was_s + ((rest - was_w) / span if span else 0.0) * (s - was_s)
            was_w, was_s = w, s
        return base + self.seconds


class Talker:
    """One participant's microphone, paced by the wall clock.

    Audio is never dropped and never sent early: chunk k is due at start + k*chunk,
    which is what a microphone does. If the service acks slower than that the
    unsent audio piles up here, and that is real — the transport admits one
    outstanding chunk per session. At the session ceiling it closes and reopens,
    because that is what a client in a long meeting has to do.
    """

    def __init__(self, url: str, pcm: bytes, sheet: Sheet, chunk: int):
        self.at = urlparse(url)
        self.pcm, self.sheet, self.chunk = pcm, sheet, chunk
        self.log: list[dict] = []
        self.shuts: list[float] = []
        self.fail: str | None = None
        self.text = ""

    def _open(self, c) -> str:
        c.request("POST", "/v1/audio/transcript",
                  json.dumps({"model": "whisper", "language": "en"}),
                  {"content-type": "application/json"})
        r = c.getresponse()
        body = r.read()
        if r.status != 201:
            raise RuntimeError(f"open {r.status}: {body[:200]!r}")
        return json.loads(body)["id"]

    def _shut(self, c, tid: str) -> dict:
        at = time.monotonic()
        c.request("DELETE", f"/v1/audio/transcript/{tid}")
        r = c.getresponse()
        body = r.read()
        self.shuts.append(time.monotonic() - at)
        return json.loads(body) if r.status == 200 else {}

    def run(self, minutes: float, begin: float) -> None:
        try:
            self._run(minutes, begin)
        except Exception as e:  # a dead talker must be visible, not silently absent
            self.fail = f"{type(e).__name__}: {e}"

    def _run(self, minutes: float, begin: float) -> None:
        c = http.client.HTTPConnection(self.at.hostname, self.at.port, timeout=1800)
        tid = self._open(c)
        said, cut, k = 0.0, 0, 0  # said: seconds this talker sent across all sessions
        time.sleep(max(0.0, begin - time.monotonic()))
        start = time.monotonic()
        while time.monotonic() - start < minutes * 60:
            time.sleep(max(0.0, start + k * self.chunk / stt.SECOND - time.monotonic()))
            piece = self.pcm[cut:cut + self.chunk]
            if len(piece) < self.chunk:
                cut, piece = 0, self.pcm[:self.chunk]
            sent = time.monotonic()
            c.request("POST", f"/v1/audio/transcript/{tid}", piece,
                      {"content-type": "application/octet-stream"})
            r = c.getresponse()
            body = r.read()
            ack = time.monotonic() - sent
            if r.status == 409:  # at the session ceiling: close, reopen, keep talking
                done = self._shut(c, tid)
                said += float(done.get("seconds", 0.0))
                self.text += " " + done.get("text", "")
                tid = self._open(c)
                continue
            if r.status != 200:
                raise RuntimeError(f"push {r.status}: {body[:200]!r}")
            state = json.loads(body)
            cut, k = cut + self.chunk, k + 1
            heard = len(WORD.findall(
                (self.text + state["text"] + " " + state["pending"]).lower()))
            self.log.append({
                "t": round(time.monotonic() - start, 3),
                "sent_s": round(said + state["seconds"], 3),
                "heard_s": round(self.sheet.at(heard), 3),
                "ack_ms": round(ack * 1000, 2),
            })
        done = self._shut(c, tid)
        self.text += " " + done.get("text", "")
        self.log.append({
            "t": round(time.monotonic() - start, 3),
            "sent_s": round(said + float(done.get("seconds", 0.0)), 3),
            "heard_s": round(self.sheet.at(len(WORD.findall(self.text.lower()))), 3),
            "shut": True,
        })
        c.close()


def slope(log: list, since: float) -> float:
    """Seconds of lag added per minute of meeting, over the tail of the run.

    Live samples only: closing flushes the whole remaining window, so the last
    point is a different quantity and would flatter a session that never kept up.
    """
    pts = [(p["t"], p["t"] - p["heard_s"]) for p in log
           if p["t"] >= since and not p.get("shut")]
    if len(pts) < 4:
        return float("nan")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    return 60.0 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else float("nan")


def talk(url: str, corpus: str, sessions: int, minutes: float, chunk_ms: int, out: str) -> None:
    sheet = Sheet.read(corpus)
    pcm = open(corpus + ".pcm", "rb").read()
    chunk = chunk_ms * stt.SECOND // 1000 // stt.WIDTH * stt.WIDTH
    live = [Talker(url, pcm, sheet, chunk) for _ in range(sessions)]
    # Staggered, so passes do not all fall due on the same tick.
    begin = time.monotonic() + 2.0
    hands = [threading.Thread(target=t.run, args=(minutes, begin + i * 0.3))
             for i, t in enumerate(live)]
    for h in hands:
        h.start()
    for h in hands:
        h.join()

    rows = []
    for i, t in enumerate(live):
        alive = [p for p in t.log if not p.get("shut")]
        if t.fail or not alive:
            rows.append({"session": i, "fail": t.fail or "no acks"})
            continue
        last, shut = alive[-1], t.log[-1]
        rows.append({
            "session": i,
            "sent_s": last["sent_s"],
            "heard_s": last["heard_s"],
            "submit_lag_s": round(last["t"] - last["sent_s"], 1),
            "decode_lag_s": round(last["sent_s"] - last["heard_s"], 1),
            "lag_s": round(last["t"] - last["heard_s"], 1),
            "lag_per_min": round(slope(t.log, last["t"] / 2), 2),
            "ack_ms_p50": round(statistics.median(p["ack_ms"] for p in alive), 2),
            # What closing recovered, and what waiting for it cost.
            "after_shut_lag_s": round(shut["t"] - shut["heard_s"], 1),
            "shut_s": round(t.shuts[-1], 1) if t.shuts else None,
            "reopened": max(0, len(t.shuts) - 1),
            "silent": last["heard_s"] == 0.0,
        })
    ok = [r for r in rows if "fail" not in r]
    said = {
        "sessions": sessions,
        "minutes": minutes,
        "per_session": rows,
        "median_lag_s": round(statistics.median([r["lag_s"] for r in ok]), 1) if ok else None,
        "worst_lag_s": max((r["lag_s"] for r in ok), default=None),
        "worst_lag_per_min": max((r["lag_per_min"] for r in ok), default=None),
        "silent": sum(r["silent"] for r in ok),
        "failed": len(rows) - len(ok),
    }
    print(json.dumps(said, indent=1))
    if out:
        json.dump({"said": said, "log": [t.log for t in live]}, open(out, "w"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    verb = ap.add_subparsers(dest="verb", required=True)

    v = verb.add_parser("say")
    v.add_argument("--voice", default="af_heart")
    v.add_argument("--out", default="say")

    v = verb.add_parser("cost")
    v.add_argument("--corpus", default="say")
    v.add_argument("--reps", type=int, default=3)
    v.add_argument("--threads", type=int, default=1)
    v.add_argument("--spans", default="1 4 8 12 16 24 30 45")

    v = verb.add_parser("talk")
    v.add_argument("--url", default="http://127.0.0.1:8000")
    v.add_argument("--corpus", default="say")
    v.add_argument("--sessions", type=int, default=1)
    v.add_argument("--minutes", type=float, default=8.0)
    v.add_argument("--chunk-ms", type=int, default=250)
    v.add_argument("--out", default="")
    a = ap.parse_args()

    if a.verb == "say":
        say(a.voice, a.out)
    elif a.verb == "cost":
        cost(a.corpus, a.reps, a.threads, [int(s) for s in a.spans.split()])
    else:
        talk(a.url, a.corpus, a.sessions, a.minutes, a.chunk_ms, a.out)


if __name__ == "__main__":
    sys.exit(main())

"""Voice activity: the audio worth decoding, and nothing else.

Silence costs a decoder exactly what speech costs it. A second of an empty room
occupies a worker for a second and bills a second, and returns "". A meeting is
mostly that — the pauses inside one person's turn, and the longer stretches
where nobody has the floor at all — so the cheapest decode is the one that never
runs.

Silero says how much each 32 ms of audio sounds like a voice. faster-whisper
already carries that model and already runs it inside every decode, so this
costs no new weight and no new bake step; what is new is running it BEFORE the
decoder instead of inside it, where the answer can save the decode rather than
merely trim its input.

A probability is not yet a decision. Deciding per window clips speech at both
ends, and a clipped first syllable costs more than the CPU it saved: a word that
begins softly is not detected until it is already under way, and a breath inside
a sentence is not the end of the sentence. So the decision is a squelch, in the
sense radio has always meant — it opens on voice, closes only after a run of
quiet, hands back the audio from just BEFORE it opened, and keeps passing for a
moment after the voice stops.
"""

import numpy
from faster_whisper.vad import get_vad_model

import stt

# Samples silero scores at once: 32 ms at 16 kHz. Its own window, not a choice.
STEP = 512

# Open loud, close quiet — the two thresholds are what stops the decision
# chattering on a syllable's own envelope. Silero's own filter reads its
# probabilities the same way, and this is the same distance apart.
OPEN = 0.5
SHUT = 0.35

# Quiet this long ends the utterance. Longer than a breath or the gap between
# words, shorter than the pause between one person finishing and the next
# starting — that is the whole span in which this number is the right one.
HANG = 0.8

# Audio held back while quiet, and handed over on the push that opens. Whisper
# reads a span, and the span it is given must contain the onset that triggered
# the decision — the trigger necessarily lags the onset, so the onset is in the
# past by the time anything knows to keep it.
LEAD = 0.5


class Squelch:
    """Passes voice, drops silence. One per session: it remembers.

    A push is admitted whole or not at all. Cutting on the sample would splice
    the window at word boundaries and hand the decoder audio with steps in it;
    admitting whole pushes keeps what the decoder reads continuous, and LEAD
    covers the onset far better than a sample-exact edge would.
    """

    def __init__(self):
        self._open = False
        self._quiet = 0.0
        self._lead = b""

    def admit(self, pcm: bytes) -> bytes:
        """Return the audio worth decoding: b"" while the room is quiet, the push
        when it is not, and the lead-in as well on the push that opens.

        A push carrying no audio carries no decision either, and silero cannot be
        asked about nothing — it indexes the last window of what it is given.
        """
        if not pcm:
            return b""

        heard = False
        for voiced in self._voiced(pcm):
            if voiced >= OPEN:
                self._open, self._quiet = True, 0.0
            elif self._open:
                self._quiet = self._quiet + STEP / stt.RATE if voiced < SHUT else 0.0
                self._open = self._quiet < HANG
            heard = heard or self._open

        if not heard:
            # Sliced from the front, because the end of a buffer is counted from
            # the end and `[-0:]` is the whole of it: written the other way, a
            # LEAD of zero keeps every second of silence instead of none, and
            # hands the lot to the decoder on the next word.
            held = self._lead + pcm
            self._lead = held[max(0, len(held) - int(LEAD * stt.SECOND)):]
            return b""
        was, self._lead = self._lead, b""
        return was + pcm

    def _voiced(self, pcm: bytes) -> numpy.ndarray:
        """One probability per STEP samples. The model wants whole windows, so a
        push that is not a whole number of them is padded — a 250 ms push at this
        rate is exactly eight, and pads not at all."""
        heard = stt.samples(pcm)
        short = -len(heard) % STEP
        if short:
            heard = numpy.pad(heard, (0, short))
        return get_vad_model()(heard).reshape(-1)

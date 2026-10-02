package main

// A transcript that grows: audio arrives in pieces, text comes back on the ack.
//
// Neither recognizer is a streaming model — both decode a span, not a sample —
// so a growing transcript is a window that is re-decoded as it fills, and the
// art is deciding which words are settled. A stretch of speech that ended a
// guard-length before the window does will not change when more audio arrives:
// it is decoded once more on its own, committed, and its audio freed. The tail
// stays open and is decoded again next time. That keeps the window bounded and
// the committed text stable, and every second of speech is committed by exactly
// one decode.
//
// The ack never waits for a decode. A push appends its audio and returns the
// newest text the session has, so the answer costs a copy and the round trip;
// decode latency lands on a later ack. One pass runs per session at a time, and
// its decodes queue behind every other session's in the recognizer's FIFO.
//
// Audio reaches the window through a squelch, so a quiet room reaches nothing:
// no samples, no decode, no charge. `seconds` is arithmetic on the samples that
// got through, and those are exactly the samples a decoder read — so the bill is
// the audio we listened to, it never exceeds the audio submitted, and it cannot
// silently become zero for audio that was decoded.

import (
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"sync"
	"time"
)

// 256 ms per push at 16 kHz mono int16. A request crosses the ai plane as ONE
// frame, so a chunk is sized to fit in one.
const (
	Chunk   = 8 * 1024
	Ceiling = 64 * 1024 // 2 s; a larger push is refused rather than truncated

	Floor  = 1.0   // audio below this is not worth a decode pass
	Guard  = 1.0   // trailing seconds never committed: the last word may still be moving
	Window = 12.0  // past this the guard is dropped, so the window cannot grow forever
	Idle   = 30.0  // a session untouched this long is collectable
	Limit  = 600.0 // audio one session will accept, in seconds
)

// The squelch. Silero opens on a window above Open and holds through anything
// above Open-0.15; quiet for Hang closes it. Lead is audio held back while the
// room is quiet and handed over by the push that opens: the trigger necessarily
// lags the onset, so the onset is in the past by the time anything knows to keep
// it, and a squelch that begins at the trigger has already eaten the first
// syllable. Measured on the Python build with LEAD = 0, 8 of 32 alignments of a
// word across a push clipped up to 71.8 ms; at 0.5 s none did.
const (
	Open = 0.5
	Hang = 0.8
	Lead = 0.5
)

// squelch passes voice and drops silence. One per session: it remembers.
//
// A push is admitted whole or not at all. Cutting on the sample would splice
// the window and hand the decoder audio with steps in it; admitting whole pushes
// keeps what the decoder reads continuous, and Lead covers the onset.
type squelch struct {
	g    gate
	lead []float32
}

// admit returns the audio worth decoding: nothing while the room is quiet, the
// push when it is not, and the lead-in as well on the push that opens.
func (s *squelch) admit(pcm []float32) []float32 {
	if len(pcm) == 0 {
		return nil
	}
	if !s.g.voiced(pcm) {
		held := append(s.lead, pcm...)
		keep := int(Lead * Rate)
		s.lead = append([]float32(nil), held[max(0, len(held)-keep):]...)
		return nil
	}
	out := append(s.lead, pcm...)
	s.lead = nil
	return out
}

// transcript is one growing transcript. Everything mutable is under mu.
type transcript struct {
	id    string
	model string // what the caller asked for
	sp    *speech

	mu       sync.Mutex
	text     string // committed: settled, never revised
	pending  string // the open tail: newest decode, may still change
	seconds  float64
	window   []float32
	got      int // samples ever admitted; marks progress for the pass
	sq       *squelch
	decoding bool
	closed   bool // no more passes; close is committing the rest
	ended    bool // the squelch's detector is freed; no more audio
	touched  time.Time

	// The ear this session decodes with, and its language. A parakeet session
	// that names no language decides at its first Hearing seconds and keeps the
	// answer; until then ear is nil.
	ear  ear
	name string
	lang string
}

func newTranscript(sp *speech, model, lang string) *transcript {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	t := &transcript{
		id: "atr_" + hex.EncodeToString(b), model: model, sp: sp,
		sq: &squelch{g: sp.vad.gate()}, touched: time.Now(),
	}
	if model == "whisper" || lang != "" {
		t.ear, t.name, t.lang, _ = sp.route(model, lang, nil)
	}
	return t
}

func (t *transcript) full() bool { return t.seconds >= Limit }

func (t *transcript) stale(now time.Time) bool {
	return now.Sub(t.touched) > time.Duration(Idle*float64(time.Second))
}

// floor is the audio a pass needs: Floor, or Hearing for a session that has yet
// to hear which language it is in.
func (t *transcript) floor() int {
	if t.ear == nil {
		return int(Hearing * Rate)
	}
	return int(Floor * Rate)
}

// push accepts audio and returns the seconds THIS push put in front of a
// decoder — the metered quantity for this call. A push the squelch drops returns
// 0. A push that OPENS it carries the lead-in and so reports more than its own
// length; those seconds were withheld from every earlier push, so the session
// still bills each second at most once and never one it did not decode.
func (t *transcript) push(pcm []float32) float64 {
	t.mu.Lock()
	if t.ended {
		t.mu.Unlock()
		return 0
	}
	t.touched = time.Now()
	heard := t.sq.admit(pcm)
	if len(heard) == 0 {
		t.mu.Unlock()
		return 0
	}
	t.window = append(t.window, heard...)
	t.got += len(heard)
	t.seconds += float64(len(heard)) / Rate
	start := !t.decoding && len(t.window) >= t.floor()
	if start {
		t.decoding = true
	}
	t.mu.Unlock()
	if start {
		go t.work()
	}
	return float64(len(heard)) / Rate
}

// work runs passes until the window stops growing. Each pass's decodes join the
// back of the recognizer's queue, so a session that keeps talking takes its turn
// behind the others rather than holding the decoder.
func (t *transcript) work() {
	for {
		t.mu.Lock()
		if t.closed || len(t.window) < t.floor() {
			t.decoding = false
			t.mu.Unlock()
			return
		}
		window, mark := append([]float32(nil), t.window...), t.got
		t.mu.Unlock()

		err := t.absorb(window, false)

		t.mu.Lock()
		if err != nil {
			// Logged, because nothing else would say it: every push still acks with
			// the right `duration` and the transcript simply stays empty. The next
			// push starts a fresh attempt.
			slog.Error("decode failed", "transcript", t.id, "err", err)
			t.decoding = false
			t.mu.Unlock()
			return
		}
		t.decoding = !t.closed && t.got != mark
		again := t.decoding
		t.mu.Unlock()
		if !again {
			return
		}
	}
}

// resolve picks the ear for a parakeet session that named no language, from the
// audio it has heard so far. Called by the one pass that is running.
func (t *transcript) resolve(window []float32) error {
	if t.ear != nil {
		return nil
	}
	e, name, lang, err := t.sp.route(t.model, "", func() []float32 { return spoken(window, t.sp.vad.regions(window)) })
	if err != nil {
		return err
	}
	t.mu.Lock()
	t.ear, t.name, t.lang = e, name, lang
	t.mu.Unlock()
	return nil
}

// absorb decodes one snapshot of the window. Stretches that end a guard-length
// before it are settled: decoded on their own and committed, their audio freed.
// The rest is decoded as one piece and becomes the pending tail. final commits
// everything, because nothing follows a close.
func (t *transcript) absorb(window []float32, final bool) error {
	if err := t.resolve(window); err != nil {
		return err
	}
	n := len(window)
	edge := n - int(Guard*Rate)
	if final || n > int(Window*Rate) {
		edge = n // past Window, holding the tail back forever would let the window grow without bound
	}
	regions := t.sp.vad.regions(window)
	var settled, open []span
	for _, r := range regions {
		if r.end <= edge {
			settled = append(settled, r)
		} else {
			open = append(open, r)
		}
	}

	parts := stretches(settled, merges[t.name])
	next := -1
	if len(open) > 0 {
		next = open[0].start
	}
	var done string
	heardTo := 0 // the end of the audio the settled decodes heard
	for i, s := range parts {
		prev, after := -1, next
		if i > 0 {
			prev = parts[i-1].end
		}
		if i+1 < len(parts) {
			after = parts[i+1].start
		}
		lo, hi := reach(s, prev, after)
		h, err := t.ear.hear(excerpt(window, lo, hi), t.lang)
		if err != nil {
			return err
		}
		done = join(done, h.text)
		heardTo = min(hi, n)
	}

	// Where the window is cut. Everything a settled decode heard goes, its
	// trailing pad included, so no later decode hears that tail again; so does
	// what the detector heard as no one speaking, except the last Guard seconds,
	// kept in case a word is starting at the edge. With an open tail, the window
	// is kept from where reach starts it — never before the settled audio ends,
	// because reach stops both at the midpoint between them.
	cut := max(0, edge, heardTo)
	pending := ""
	if len(open) > 0 {
		prev := -1
		if len(parts) > 0 {
			prev = parts[len(parts)-1].end
		}
		lo, _ := reach(span{open[0].start, n}, prev, -1)
		cut = max(0, lo)
		h, err := t.ear.hear(excerpt(window, lo, n+int(Pad*Rate)), t.lang)
		if err != nil {
			return err
		}
		pending = h.text
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed && !final {
		return nil // close commits the remainder itself
	}
	t.text = join(t.text, done)
	t.window = t.window[min(cut, len(t.window)):]
	t.pending = pending
	return nil
}

// close decodes what is left and commits all of it. Blocks: the final text is
// the point of closing. It waits out a pass already running, so the window it
// commits is the one that pass left, and nothing is committed twice.
func (t *transcript) close() error {
	t.mu.Lock()
	t.closed = true
	t.mu.Unlock()
	for {
		t.mu.Lock()
		busy := t.decoding
		rest := append([]float32(nil), t.window...)
		t.mu.Unlock()
		if !busy {
			defer t.end()
			if len(rest) > 0 {
				if err := t.absorb(rest, true); err != nil {
					return err
				}
			}
			t.mu.Lock()
			t.window, t.pending = nil, ""
			t.mu.Unlock()
			return nil
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// end frees the session's squelch detector. Under the lock, so no push is
// inside it when it goes.
func (t *transcript) end() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.ended {
		t.ended = true
		t.sq.g.close()
	}
}

// state is what every ack carries. `duration` is what THIS call consumed, named
// as the batch route names it, so the plane meters a push and a batch
// transcription by one rule; `seconds` is the session's running total.
type state struct {
	ID       string  `json:"id"`
	Text     string  `json:"text"`
	Pending  string  `json:"pending"`
	Seconds  float64 `json:"seconds"`
	Duration float64 `json:"duration"`
}

func (t *transcript) state(billed float64) state {
	t.mu.Lock()
	defer t.mu.Unlock()
	return state{ID: t.id, Text: t.text, Pending: t.pending, Seconds: round(t.seconds), Duration: round(billed)}
}

// sessions holds this process's open transcripts. A growing transcript is a
// window in ONE process's memory, which is why `open` says where it lives.
type sessions struct {
	mu   sync.Mutex
	live map[string]*transcript
}

// begin opens a transcript, collecting abandoned ones on the way in: sessions
// are only worth collecting when new ones arrive, so there is no clock to run
// and no task to supervise.
func (ss *sessions) begin(t *transcript) {
	now := time.Now()
	ss.mu.Lock()
	defer ss.mu.Unlock()
	if ss.live == nil {
		ss.live = map[string]*transcript{}
	}
	for id, old := range ss.live {
		old.mu.Lock()
		stale := old.stale(now)
		old.mu.Unlock()
		if stale {
			delete(ss.live, id)
			old.mu.Lock()
			old.closed = true
			old.mu.Unlock()
			go old.end()
		}
	}
	ss.live[t.id] = t
}

func (ss *sessions) find(id string) *transcript {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	return ss.live[id]
}

func (ss *sessions) drop(id string) *transcript {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	t := ss.live[id]
	delete(ss.live, id)
	return t
}

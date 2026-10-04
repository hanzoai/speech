package main

import (
	"fmt"
	"log/slog"
	"strings"
)

// Rates the models actually work at. Neither is a choice: every recognizer here
// reads 16 kHz, and both synthesizers speak 24 kHz.
const (
	Rate   = 16000        // what a recognizer hears
	Width  = 2            // bytes per pcm16 sample
	Second = Rate * Width // bytes of raw pcm16 per second
	Spoken = 24000        // what a synthesizer says
)

// How audio is cut into the spans a recognizer decodes. A span is speech
// between pauses, so a cut never lands inside a word, and whisper's spans stay
// under the thirty seconds it can attend over.
const (
	Pause   = 0.3  // quiet this long ends a stretch of speech
	Pad     = 0.5  // audio kept either side of a stretch, for its onset and its tail
	Gap     = 2.0  // whisper decodes stretches closer than this together
	Longest = 20.0 // and merges none past this length
	Hardest = 28.0 // and decodes none longer than this: its ceiling is 30
	Search  = 3.0  // a cut that must fall inside speech looks this far back for a lull
	Lull    = 0.08 // the stretch of audio a cut point is weighed by
)

// How a parakeet request that names no language finds out which one it is.
const (
	Hearing = 2.0  // audio a growing transcript waits for before it asks
	Listen  = 30.0 // the most audio language identification reads
)

// word is one word and when it was said, in seconds from the start of the audio
// the caller sent — what a caption cuts on.
type word struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// heard is one decode: its text and, when the model times its tokens, its words
// in seconds from the start of the span it was handed.
type heard struct {
	text  string
	words []word
}

// An ear turns 16 kHz mono samples into words. lang is a code the ear's model
// knows, or "" to let it decide.
type ear interface {
	hear(pcm []float32, lang string) (heard, error)
}

// A lid says which language is being spoken.
type lid interface {
	identify(pcm []float32) (string, error)
}

// A mouth turns text into 24 kHz mono samples in one of its voices.
type mouth interface {
	speak(text, voice string, speed float32) ([]float32, error)
}

// A gate is one session's squelch detector: it remembers the room across pushes.
type gate interface {
	voiced(pcm []float32) bool
	close()
}

// span is a stretch of samples, [start, end).
type span struct{ start, end int }

func (s span) seconds() (float64, float64) { return float64(s.start) / Rate, float64(s.end) / Rate }

// A vad finds speech: a gate per session, and the stretches of speech in a span.
type vad interface {
	gate() gate
	regions(pcm []float32) []span
}

// speech is every model the service serves, behind the interfaces above.
type speech struct {
	ears   map[string]ear   // "parakeet", "whisper"
	lid    lid              // routes parakeet's foreign speech to whisper
	mouths map[string]mouth // "kokoro"
	vad    vad
}

// heard and spoken list the model ids, in the order /v1/models names them.
var (
	heardBy  = []string{"parakeet", "whisper"}
	spokenBy = []string{"kokoro"}
)

func (sp *speech) models() []string {
	var out []string
	for _, m := range heardBy {
		if sp.ears[m] != nil {
			out = append(out, m)
		}
	}
	for _, m := range spokenBy {
		if sp.mouths[m] != nil {
			out = append(out, m)
		}
	}
	return out
}

// route decides which ear decodes a request, and in which language.
//
// whisper decodes everything, in the language named or the one it detects.
// parakeet decodes its 25 languages and hands the rest to whisper: a language
// named outside its set goes straight there, and with none named the audio is
// identified first. heard is the speech identification reads — the caller's
// audio with its silences cut out, at most Listen seconds of it — and is read
// only when it has to be.
//
// The language handed to whisper on a fallback is "" when it was identified
// rather than named: whisper's own detection runs on a model forty times the
// size of the identifier's, and it is one decoder step.
func (sp *speech) route(model, lang string, heard func() []float32) (ear, string, string, error) {
	switch model {
	case "whisper":
		return sp.ears["whisper"], "whisper", lang, nil
	case "parakeet":
		if lang != "" {
			if parakeetLanguages[lang] {
				return sp.ears["parakeet"], "parakeet", lang, nil
			}
			return sp.ears["whisper"], "whisper", lang, nil
		}
		pcm := heard()
		if len(pcm) == 0 {
			return sp.ears["parakeet"], "parakeet", "", nil
		}
		said, err := sp.lid.identify(pcm)
		if err != nil {
			return nil, "", "", err
		}
		if parakeetLanguages[said] {
			return sp.ears["parakeet"], "parakeet", "", nil
		}
		slog.Info("parakeet hands a language it lacks to whisper", "identified", said)
		return sp.ears["whisper"], "whisper", "", nil
	}
	return nil, "", "", fmt.Errorf("model %q transcribes nothing", model)
}

// merges says which recognizers decode neighbouring stretches of speech as one
// span. whisper does: it attends over the whole span, so the sentence before
// is context for the sentence after, and each decode carries a fixed cost
// (about half a second on four cores) that merging pays once. Parakeet does
// not: measured on its int8 export, a span holding two speakers with room tone
// between them lost the second speaker in 24 of 64 trials; decoded stretch by
// stretch, with a stretch ending at Pause of quiet, none of 72 did.
var merges = map[string]bool{"whisper": true}

// stretches cuts audio into the spans that are decoded: speech regions, merged
// while they are close together and short enough when the recognizer merges,
// and split by length if one is still too long for whisper. A split lands where
// quiet says, which is the quietest moment before the length runs out: cut
// mid-word, the decode on either side answers with the whole word.
func stretches(regions []span, merge bool, quiet func(lo, hi int) int) []span {
	var out []span
	for _, r := range regions {
		if n := len(out); merge && n > 0 {
			last := &out[n-1]
			if float64(r.start-last.end) < Gap*Rate && float64(r.end-last.start) <= Longest*Rate {
				last.end = r.end
				continue
			}
		}
		out = append(out, r)
	}
	var cut []span
	hard := int(Hardest * Rate)
	for _, s := range out {
		for s.end-s.start > hard {
			at := quiet(s.start+hard-int(Search*Rate), s.start+hard)
			cut = append(cut, span{s.start, at})
			s.start = at
		}
		cut = append(cut, s)
	}
	return cut
}

// quietest is where to cut pcm somewhere in [lo, hi): the middle of the quietest
// Lull of it, weighed in 10 ms steps. A recognizer handed half a word answers
// with a word, so the half before a cut and the half after both come back whole
// — "music to to children", "a tiny cab. Cafe". Between words, or at a comma,
// there is nothing to hear twice.
func quietest(pcm []float32, lo, hi int) int {
	lo, hi = max(lo, 0), min(hi, len(pcm))
	lull, step := int(Lull*Rate), Rate/100
	if hi-lo <= lull {
		return hi
	}
	energy := func(a int) float64 {
		var e float64
		for _, x := range pcm[a : a+lull] {
			e += float64(x) * float64(x)
		}
		return e
	}
	best, at := energy(lo), lo
	for a := lo + step; a+lull <= hi; a += step {
		if e := energy(a); e <= best { // the latest of equals: the more is committed, the smaller the window
			best, at = e, a
		}
	}
	return at + lull/2
}

// split cuts any region that runs across at into the part before it and the part
// after, so a cut that has to fall inside speech still has spans to decode on
// either side of it.
func split(regions []span, at int) []span {
	out := make([]span, 0, len(regions)+1)
	for _, r := range regions {
		if r.start < at && at < r.end {
			out = append(out, span{r.start, at}, span{at, r.end})
			continue
		}
		out = append(out, r)
	}
	return out
}

// reach is the audio a stretch is decoded from: Pad either side of its speech,
// but never past the midpoint to a neighbouring stretch, so no word is heard
// twice and no other speaker's tail rides in. prev and next are the stretch
// ends either side, or -1 where there is none.
func reach(s span, prev, next int) (int, int) {
	pad := int(Pad * Rate)
	lo, hi := s.start-pad, s.end+pad
	if prev >= 0 {
		lo = max(lo, (prev+s.start)/2)
	}
	if next >= 0 {
		hi = min(hi, (s.end+next)/2)
	}
	return lo, hi
}

// excerpt is pcm[lo:hi] with silence wherever the range runs past the audio.
// Every decode is handed Pad of audio either side of its speech, and a word at
// the very first sample still needs that lead-in: decoded with none, "The quick
// brown fox" came back as "Quick brown fox" and "A quick brown fox".
func excerpt(pcm []float32, lo, hi int) []float32 {
	out := make([]float32, hi-lo)
	a, b := max(lo, 0), min(hi, len(pcm))
	if a < b {
		copy(out[a-lo:], pcm[a:b])
	}
	return out
}

// spoken is the audio of the stretches, end to end, up to Listen seconds: what
// language identification reads, with the room's silences left out.
func spoken(pcm []float32, parts []span) []float32 {
	limit := int(Listen * Rate)
	var out []float32
	for _, s := range parts {
		out = append(out, pcm[s.start:s.end]...)
		if len(out) >= limit {
			return out[:limit]
		}
	}
	return out
}

// segment is one decoded stretch of a batch transcription, shaped as OpenAI's
// verbose_json segment for the fields this decoder actually knows.
type segment struct {
	ID    int     `json:"id"`
	Seek  int     `json:"seek"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// transcription is one batch transcription: the text, the length of the audio
// submitted (the billable quantity), and where every stretch and word fell.
type transcription struct {
	text     string
	duration float64
	segments []segment
	words    []word
}

// transcribe decodes a whole recording.
//
// duration is the audio SUBMITTED, not the speech found in it: a caller is
// charged for what they asked us to listen to, not for how much of it turned
// out to be talking — the same rule the Python service billed by.
func (sp *speech) transcribe(model, lang string, pcm []float32) (transcription, error) {
	out := transcription{duration: float64(len(pcm)) / Rate}
	regions := sp.vad.regions(pcm)
	if len(regions) == 0 {
		return out, nil
	}
	e, engine, code, err := sp.route(model, lang, func() []float32 { return spoken(pcm, regions) })
	if err != nil {
		return out, err
	}
	parts := stretches(regions, merges[engine], func(lo, hi int) int { return quietest(pcm, lo, hi) })
	for i, s := range parts {
		prev, next := -1, -1
		if i > 0 {
			prev = parts[i-1].end
		}
		if i+1 < len(parts) {
			next = parts[i+1].start
		}
		lo, hi := reach(s, prev, next)
		h, err := e.hear(excerpt(pcm, lo, hi), code)
		if err != nil {
			return out, fmt.Errorf("%s: %w", engine, err)
		}
		start, end := s.seconds()
		out.segments = append(out.segments, segment{ID: i, Seek: s.start / (Rate / 100), Start: round(start), End: round(end), Text: h.text})
		at := float64(lo) / Rate
		for _, w := range h.words {
			out.words = append(out.words, word{Word: w.Word, Start: round(clamp(at+w.Start, out.duration)), End: round(clamp(at+w.End, out.duration))})
		}
		out.text = join(out.text, h.text)
	}
	return out, nil
}

// join puts two pieces of text together with a space, unless both sides of the
// seam are a script written without one.
func join(a, b string) string {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	ra, rb := []rune(a), []rune(b)
	if unspaced(string(ra[len(ra)-1])) && unspaced(string(rb[0])) {
		return a + b
	}
	return a + " " + b
}

func clamp(x, top float64) float64 { return max(0, min(x, top)) }

func round(x float64) float64 {
	const k = 1000
	if x < 0 {
		return -round(-x)
	}
	return float64(int64(x*k+0.5)) / k
}

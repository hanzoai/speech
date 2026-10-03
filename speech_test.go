package main

// The service's contract, tested without a weight on disk.
//
// The stand-ins replace the MODELS — the ear, the detector, the identifier, the
// mouth — never the code under test, so the window arithmetic, the squelch, the
// routing and the HTTP shapes are the real ones. The ear's stand-in names every
// tone it is handed: audio that says its own name, so a cut that drops the wrong
// samples, too many or none at all does not merely change a length here — it
// changes the words.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── stand-ins ────────────────────────────────────────────────────────────────

// tone is audio whose every sample is value/32768: a span that says its name.
func tone(value int, seconds float64) []float32 {
	out := make([]float32, int(seconds*Rate))
	for i := range out {
		out[i] = float32(value) / 32768
	}
	return out
}

func quiet(seconds float64) []float32 { return tone(0, seconds) }

func cat(parts ...[]float32) []float32 {
	var out []float32
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// namingEar hears the distinct tones in what it is handed, in order. It records
// every call so a test can see what was decoded, and in which language.
type namingEar struct {
	mu    sync.Mutex
	spans []float64
	langs []string
	slow  time.Duration
	fail  error
}

func (e *namingEar) hear(pcm []float32, lang string) (heard, error) {
	e.mu.Lock()
	e.spans = append(e.spans, float64(len(pcm))/Rate)
	e.langs = append(e.langs, lang)
	e.mu.Unlock()
	if e.slow > 0 {
		time.Sleep(e.slow)
	}
	if e.fail != nil {
		return heard{}, e.fail
	}
	var names []string
	var ws []word
	last := 0
	for i, x := range pcm {
		v := int(math.Round(float64(x) * 32768))
		if v == 0 || v == last {
			if v == 0 {
				last = 0
			}
			continue
		}
		last = v
		names = append(names, strconv.Itoa(v))
		ws = append(ws, word{Word: strconv.Itoa(v), Start: float64(i) / Rate, End: float64(i)/Rate + 0.1})
	}
	return heard{text: strings.Join(names, " "), words: ws}, nil
}

func (e *namingEar) calls() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.spans)
}

// roomVAD finds speech by amplitude: any nonzero sample is speech for the
// segmenter, and a sample above half scale opens the gate. The gate is the same
// two-threshold squelch silero runs, window by window, so its hangover is real
// arithmetic rather than a constant.
type roomVAD struct{}

func (roomVAD) gate() gate { return &roomGate{} }

func (roomVAD) regions(pcm []float32) []span {
	var out []span
	start := -1
	gap := int(Pause * Rate)
	quietRun := 0
	for i, x := range pcm {
		if x != 0 {
			if start < 0 {
				start = i
			}
			quietRun = 0
			continue
		}
		if start >= 0 {
			quietRun++
			if quietRun >= gap {
				out = append(out, span{start, i - quietRun + 1})
				start, quietRun = -1, 0
			}
		}
	}
	if start >= 0 {
		out = append(out, span{start, len(pcm) - quietRun})
	}
	return out
}

type roomGate struct {
	open   bool
	quiet  float64
	closed bool
}

func (g *roomGate) voiced(pcm []float32) bool {
	heard := false
	for at := 0; at < len(pcm); at += window {
		loud := false
		for _, x := range pcm[at:min(at+window, len(pcm))] {
			if math.Abs(float64(x)) >= Open {
				loud = true
				break
			}
		}
		if loud {
			g.open, g.quiet = true, 0
		} else if g.open {
			g.quiet += float64(window) / Rate
			g.open = g.quiet < Hang
		}
		heard = heard || g.open
	}
	return heard
}

func (g *roomGate) close() { g.closed = true }

// alwaysVAD hears a voice in everything: for tests about the window, not the room.
type alwaysVAD struct{ roomVAD }

func (alwaysVAD) gate() gate { return alwaysGate{} }

type alwaysGate struct{}

func (alwaysGate) voiced([]float32) bool { return true }
func (alwaysGate) close()                {}

type fixedLID struct{ lang string }

func (l fixedLID) identify([]float32) (string, error) { return l.lang, nil }

// toneMouth speaks a 440 Hz tone, a tenth of a second per character, and
// remembers how it was asked.
type toneMouth struct {
	voice string
	speed float32
}

func (m *toneMouth) speak(text, voice string, speed float32) ([]float32, error) {
	m.voice, m.speed = voice, speed
	n := int(float64(len(text)) * 0.1 * Spoken / float64(speed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.3 * math.Sin(2*math.Pi*440*float64(i)/Spoken))
	}
	return out, nil
}

type rig struct {
	t        *testing.T
	s        *server
	srv      *httptest.Server
	parakeet *namingEar
	whisper  *namingEar
	mouth    *toneMouth
	sp       *speech
}

func newRig(t *testing.T, v vad, lid string) *rig {
	t.Helper()
	r := &rig{t: t, parakeet: &namingEar{}, whisper: &namingEar{}, mouth: &toneMouth{}}
	r.sp = &speech{
		ears:   map[string]ear{"parakeet": r.parakeet, "whisper": r.whisper},
		lid:    fixedLID{lid},
		mouths: map[string]mouth{"kokoro": r.mouth},
		vad:    v,
	}
	r.s = &server{maxIn: 100 << 20}
	r.s.sp.Store(r.sp)
	r.srv = httptest.NewServer(r.s.routes())
	t.Cleanup(r.srv.Close)
	return r
}

func wavOf(pcm []float32) []byte { return wav(pcm, Rate) }

// transcribeForm posts a multipart transcription.
func (r *rig) transcribe(file []byte, fields map[string][]string) (*http.Response, []byte) {
	r.t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	fw, _ := w.CreateFormFile("file", "audio.webm") // the ai plane's name for every upload
	fw.Write(file)
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		for _, v := range fields[k] {
			w.WriteField(k, v)
		}
	}
	w.Close()
	resp, err := http.Post(r.srv.URL+"/v1/audio/transcriptions", w.FormDataContentType(), &body)
	if err != nil {
		r.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func (r *rig) post(path string, body any) (*http.Response, []byte) {
	r.t.Helper()
	var rd io.Reader
	ctype := "application/json"
	switch b := body.(type) {
	case []byte:
		rd, ctype = bytes.NewReader(b), "application/octet-stream"
	case nil:
		rd = strings.NewReader("")
	default:
		j, _ := json.Marshal(b)
		rd = bytes.NewReader(j)
	}
	resp, err := http.Post(r.srv.URL+path, ctype, rd)
	if err != nil {
		r.t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, out
}

func (r *rig) delete(path string) (*http.Response, []byte) {
	req, _ := http.NewRequest(http.MethodDelete, r.srv.URL+path, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		r.t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, out
}

func decodeJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("not JSON: %s", b)
	}
	return m
}

func ffmpegOrSkip(t *testing.T) {
	t.Helper()
	if _, err := os.Stat("/usr/bin/ffmpeg"); err != nil {
		if p, _ := filepathLook("ffmpeg"); p == "" {
			t.Skip("no ffmpeg on this machine: the HTTP transcription path decodes through it")
		}
	}
}

func filepathLook(name string) (string, error) {
	for _, d := range filepath.SplitList(os.Getenv("PATH")) {
		if p := filepath.Join(d, name); fileExists(p) {
			return p, nil
		}
	}
	return "", os.ErrNotExist
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// ── batch transcription: the duration is the billable quantity ─────────────

// A recording: two utterances with a pause between them longer than Gap, so
// they are two stretches, in a quiet room.
var recording = cat(quiet(0.5), tone(7, 2.0), quiet(2.5), tone(9, 1.5), quiet(0.5))

func TestVerboseJSONCarriesTheDurationSubmitted(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "response_format": {"verbose_json"}})
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	body := decodeJSON(t, b)
	// The audio SUBMITTED, quiet included: a caller pays for what they asked us
	// to listen to, not for how much of it turned out to be talking.
	if got, want := body["duration"].(float64), 7.0; math.Abs(got-want) > 0.01 {
		t.Fatalf("duration %v, want %v — 0 is unbillable, speech-only underbills", got, want)
	}
	if body["text"] != "7 9" || body["task"] != "transcribe" {
		t.Fatalf("body %v", body)
	}
}

// A CAP ON LENGTH BOUNDS THE WORK. The ai plane's public lane holds a visitor to a
// length with max_seconds; audio past it is refused before any decode of the model,
// and the ffmpeg decode itself stops just past the cap.
func TestMaxSecondsRefusesAudioPastIt(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "max_seconds": {"5"}})
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("7 s under a 5 s cap answered %d %s, want 413", resp.StatusCode, b)
	}
	resp, b = r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "max_seconds": {"7"}, "response_format": {"verbose_json"}})
	if resp.StatusCode != 200 {
		t.Fatalf("7 s under a 7 s cap answered %d %s, want 200", resp.StatusCode, b)
	}
	if got := decodeJSON(t, b)["duration"].(float64); math.Abs(got-7) > 0.01 {
		t.Fatalf("duration %v, want 7", got)
	}
	resp, _ = r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "max_seconds": {"-1"}})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("a negative cap answered %d, want 400", resp.StatusCode)
	}
}

func TestPlainJSONIsJustTheText(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	_, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}})
	if string(bytes.TrimSpace(b)) != `{"text":"7 9"}` {
		t.Fatalf("json body must stay {text}: %s", b)
	}
}

func TestTextFormatIsPlainText(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "response_format": {"text"}})
	if string(b) != "7 9" || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/plain") {
		t.Fatalf("%q %s", b, resp.Header.Get("Content-Type"))
	}
}

func TestSegmentsAreTheDefaultAndWordsAreAskedFor(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	_, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"parakeet"}, "response_format": {"verbose_json"}})
	body := decodeJSON(t, b)
	if _, ok := body["words"]; ok {
		t.Fatal("words appear only when asked for")
	}
	segs := body["segments"].([]any)
	if len(segs) != 2 {
		t.Fatalf("two utterances further apart than Gap are two segments: %v", segs)
	}
	first := segs[0].(map[string]any)
	if first["text"] != "7" || math.Abs(first["start"].(float64)-0.5) > 0.01 || math.Abs(first["end"].(float64)-2.5) > 0.01 {
		t.Fatalf("segment 0 %v", first)
	}

	_, b = r.transcribe(wavOf(recording), map[string][]string{"model": {"parakeet"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word"}})
	body = decodeJSON(t, b)
	if _, ok := body["segments"]; ok {
		t.Fatal("word alone omits segments")
	}
	words := body["words"].([]any)
	if len(words) != 2 {
		t.Fatalf("words %v", words)
	}
	// Each word is timed from the start of the RECORDING, not of its segment.
	w0, w1 := words[0].(map[string]any), words[1].(map[string]any)
	if w0["word"] != "7" || math.Abs(w0["start"].(float64)-0.5) > 0.01 || w1["word"] != "9" || math.Abs(w1["start"].(float64)-5.0) > 0.01 {
		t.Fatalf("word timings %v %v", w0, w1)
	}

	_, b = r.transcribe(wavOf(recording), map[string][]string{"model": {"parakeet"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word", "segment"}})
	body = decodeJSON(t, b)
	if len(body["words"].([]any)) != 2 || len(body["segments"].([]any)) != 2 {
		t.Fatalf("both granularities: %v", body)
	}
}

func TestTimingsRequireVerboseJSON(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "response_format": {"json"}, "timestamp_granularities[]": {"word"}})
	if resp.StatusCode != 400 || !strings.Contains(string(b), "verbose_json") {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

func TestUnknownGranularityIsRefusedByName(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"phoneme"}})
	if resp.StatusCode != 400 || !strings.Contains(string(b), "phoneme") {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

func TestUnknownModelAndFormatAndLanguage(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	if resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"not-a-model"}}); resp.StatusCode != 404 || !strings.Contains(string(b), "parakeet") {
		t.Fatalf("unknown model: %d %s", resp.StatusCode, b)
	}
	if resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "response_format": {"srt"}}); resp.StatusCode != 400 || !strings.Contains(string(b), "srt") {
		t.Fatalf("unknown format: %d %s", resp.StatusCode, b)
	}
	if resp, b := r.transcribe(wavOf(recording), map[string][]string{"model": {"whisper"}, "language": {"klingon"}}); resp.StatusCode != 400 || !strings.Contains(string(b), "klingon") {
		t.Fatalf("unknown language: %d %s", resp.StatusCode, b)
	}
}

func TestAFileThatIsNotAudioIsA400(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.transcribe([]byte("this is not audio at all"), map[string][]string{"model": {"whisper"}})
	if resp.StatusCode != 400 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

func TestSilenceTranscribesToNothingAndStillBills(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	got, err := r.sp.transcribe("parakeet", "", quiet(3))
	if err != nil {
		t.Fatal(err)
	}
	if got.text != "" || got.duration != 3 || r.parakeet.calls() != 0 {
		t.Fatalf("silence: %+v, decodes %d", got, r.parakeet.calls())
	}
}

// ── routing: parakeet's 25 languages, whisper for the rest ────────────────────

func TestParakeetRoutesWhatItLacksToWhisper(t *testing.T) {
	cases := []struct {
		name, lang, lid string
		engine, handed  string
	}{
		{"a European language named", "de", "zh", "parakeet", "de"},
		{"a language outside the set named", "ja", "en", "whisper", "ja"},
		{"none named, identified European", "", "fr", "parakeet", ""},
		// whisper detects again, on a model forty times the identifier's size
		{"none named, identified outside", "", "zh", "whisper", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := newRig(t, roomVAD{}, c.lid)
			if _, err := r.sp.transcribe("parakeet", c.lang, recording); err != nil {
				t.Fatal(err)
			}
			ears := map[string]*namingEar{"parakeet": r.parakeet, "whisper": r.whisper}
			e := ears[c.engine]
			if e.calls() == 0 {
				t.Fatalf("%s decoded nothing", c.engine)
			}
			for _, l := range e.langs {
				if l != c.handed {
					t.Fatalf("%s was handed language %q, want %q", c.engine, l, c.handed)
				}
			}
			other := r.whisper
			if c.engine == "whisper" {
				other = r.parakeet
			}
			if other.calls() != 0 {
				t.Fatal("one request, one engine")
			}
		})
	}
}

func TestWhisperIsHandedTheLanguageAsNamed(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	for raw, want := range map[string]string{"de": "de", "en-US": "en", "PT_br": "pt", "jv": "jw", "": ""} {
		code, err := language(raw)
		if err != nil || code != want {
			t.Fatalf("language(%q) = %q, %v; want %q", raw, code, err, want)
		}
	}
	if _, err := r.sp.transcribe("whisper", "ko", recording); err != nil {
		t.Fatal(err)
	}
	if r.whisper.langs[0] != "ko" {
		t.Fatalf("whisper handed %q", r.whisper.langs[0])
	}
}

func TestEveryParakeetLanguageIsOneWhisperKnows(t *testing.T) {
	// A fallback names whisper the caller's language, so the set parakeet covers
	// must sit inside the set whisper accepts.
	for code := range parakeetLanguages {
		if !whisperLanguages[code] {
			t.Fatalf("%s is parakeet's but whisper does not know it", code)
		}
	}
	if len(parakeetLanguages) != 25 {
		t.Fatalf("Parakeet v3 covers 25 European languages, not %d", len(parakeetLanguages))
	}
}

func TestStretchesMergeForWhisperAndSplitLongSpeech(t *testing.T) {
	s := func(a, b float64) span { return span{int(a * Rate), int(b * Rate)} }
	regions := []span{s(0, 2), s(3, 5), s(8, 9)}
	if got, want := stretches(regions, true), []span{s(0, 5), s(8, 9)}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("whisper merges close speech: %v want %v", got, want)
	}
	if got := stretches(regions, false); fmt.Sprint(got) != fmt.Sprint(regions) {
		t.Fatalf("parakeet decodes each stretch on its own: %v", got)
	}
	if !merges["whisper"] || merges["parakeet"] {
		t.Fatal("whisper merges; parakeet does not")
	}
	if got := stretches([]span{s(0, 18), s(19, 30)}, true); len(got) != 2 {
		t.Fatalf("no merge past Longest: %v", got)
	}
	for _, merge := range []bool{true, false} {
		got := stretches([]span{s(0, 70)}, merge)
		for _, g := range got {
			if float64(g.end-g.start) > Hardest*Rate {
				t.Fatalf("a stretch longer than whisper can attend over: %v", got)
			}
		}
		if got[0].start != 0 || got[len(got)-1].end != int(70*Rate) {
			t.Fatalf("split must cover the stretch: %v", got)
		}
	}
}

func TestReachStopsHalfwayToTheNeighbours(t *testing.T) {
	sec := func(x float64) int { return int(x * Rate) }
	lo, hi := reach(span{sec(5), sec(7)}, -1, -1)
	if lo != sec(5-Pad) || hi != sec(7+Pad) {
		t.Fatalf("alone: Pad either side, %d %d", lo, hi)
	}
	lo, hi = reach(span{sec(5), sec(7)}, sec(4.8), sec(7.2))
	if lo != sec(4.9) || hi != sec(7.1) {
		t.Fatalf("between close neighbours: halfway, %v %v", float64(lo)/Rate, float64(hi)/Rate)
	}
	lo, _ = reach(span{0, sec(1)}, -1, -1)
	if lo >= 0 {
		t.Fatal("at the first sample the lead-in is silence, not nothing")
	}
	if got := excerpt([]float32{1, 2, 3}, -2, 5); fmt.Sprint(got) != "[0 0 1 2 3 0 0]" {
		t.Fatal(got)
	}
}

func TestWordsComeFromTimedTokens(t *testing.T) {
	// Parakeet's SentencePiece marks a word's first piece with ▁ and gives every
	// token a duration.
	ws := words([]string{"▁The", "▁qu", "ick", "▁fox", "."}, []float32{0.1, 0.4, 0.6, 0.9, 1.2}, []float32{0.2, 0.2, 0.2, 0.3, 0.1}, false)
	near := func(a, b float64) bool { return math.Abs(a-b) < 1e-6 }
	if len(ws) != 3 || ws[0].Word != "The" || ws[1].Word != "quick" || ws[2].Word != "fox." ||
		!near(ws[1].Start, 0.4) || !near(ws[1].End, 0.8) || !near(ws[2].End, 1.3) {
		t.Fatalf("parakeet words %v", ws)
	}
	// whisper's BPE marks it with a space, and its special tokens are not words.
	ws = words([]string{"<|en|>", " Hello", " wor", "ld"}, []float32{0, 0.5, 1.0, 1.2}, nil, true)
	if len(ws) != 2 || ws[0].Word != "Hello" || ws[1].Word != "world" || ws[0].End != 1.0 {
		t.Fatalf("whisper words %v", ws)
	}
	// A script without spaces: each token is a word, which is how whisper splits them.
	ws = words([]string{"今天", "天气", "很好"}, []float32{0, 0.4, 0.8}, nil, true)
	if len(ws) != 3 {
		t.Fatalf("han words %v", ws)
	}
	// No timestamps means the model timed nothing: no words rather than invented ones.
	if words([]string{" a", " b"}, nil, nil, true) != nil {
		t.Fatal("untimed tokens must not become timed words")
	}
}

func TestJoinLeavesNoSpaceBetweenHanCharacters(t *testing.T) {
	if join("今天天气很好", "我们去散步") != "今天天气很好我们去散步" || join("hello", "world") != "hello world" || join("", "x") != "x" {
		t.Fatal("join")
	}
}

// ── speech: the container is what we say it is ────────────────────────────────

func TestEveryFormatIsLabelledAndEncodedAsItself(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	magic := map[string]func([]byte) bool{
		"wav": func(b []byte) bool { return bytes.HasPrefix(b, []byte("RIFF")) },
		"mp3": func(b []byte) bool { return bytes.HasPrefix(b, []byte("ID3")) || (b[0] == 0xff && b[1]&0xe0 == 0xe0) },
		"opus": func(b []byte) bool {
			return bytes.HasPrefix(b, []byte("OggS")) && bytes.Contains(b[:100], []byte("OpusHead"))
		},
		"aac":  func(b []byte) bool { return b[0] == 0xff && b[1]&0xf0 == 0xf0 },
		"flac": func(b []byte) bool { return bytes.HasPrefix(b, []byte("fLaC")) },
		"pcm":  func(b []byte) bool { return len(b)%2 == 0 && !bytes.HasPrefix(b, []byte("RIFF")) },
	}
	mimes := map[string]string{"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/pcm"}
	for f, ok := range magic {
		resp, b := r.post("/v1/audio/speech", map[string]any{"model": "kokoro", "input": "hello there", "response_format": f})
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", f, resp.StatusCode, b)
		}
		if got := resp.Header.Get("Content-Type"); got != mimes[f] {
			t.Fatalf("%s labelled %s", f, got)
		}
		if !ok(b) {
			t.Fatalf("%s: the bytes are not %s (%x)", f, f, b[:min(16, len(b))])
		}
	}
	// pcm is the raw samples at 24 kHz: 0.1 s per character of input here.
	_, b := r.post("/v1/audio/speech", map[string]any{"input": "hello there", "response_format": "pcm"})
	text := "hello there"
	if want := int(float64(len(text))*0.1*Spoken) * 2; len(b) != want {
		t.Fatalf("pcm %d bytes, want %d", len(b), want)
	}
}

func TestFormatsCoverTheOpenAISet(t *testing.T) {
	var names []string
	for n := range formats {
		names = append(names, n)
	}
	sort.Strings(names)
	if strings.Join(names, " ") != "aac flac mp3 opus pcm wav" {
		t.Fatal(names)
	}
}

func TestAnUnsupportedFormatIsRefusedByName(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.post("/v1/audio/speech", map[string]any{"input": "hi", "response_format": "midi"})
	if resp.StatusCode != 400 || !strings.Contains(string(b), "midi") {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	for f := range formats {
		if !strings.Contains(string(b), f) {
			t.Fatalf("the refusal must name %s: %s", f, b)
		}
	}
}

func TestTheDefaultsAreMP3InAfHeart(t *testing.T) {
	ffmpegOrSkip(t)
	r := newRig(t, roomVAD{}, "en")
	resp, _ := r.post("/v1/audio/speech", map[string]any{"input": "hi"})
	if resp.Header.Get("Content-Type") != "audio/mpeg" || r.mouth.voice != "af_heart" || r.mouth.speed != 1 {
		t.Fatalf("%s %s %v", resp.Header.Get("Content-Type"), r.mouth.voice, r.mouth.speed)
	}
}

func TestVoicesByKokoroIDAndByOpenAIName(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	for asked, want := range map[string]string{"bf_emma": "bf_emma", "am_michael": "am_michael", "nova": "af_nova", "Alloy": "af_alloy", "fable": "bm_fable", "verse": "am_puck", "zf_xiaoxiao": "zf_xiaoxiao"} {
		resp, b := r.post("/v1/audio/speech", map[string]any{"input": "hi", "voice": asked, "response_format": "wav"})
		if resp.StatusCode != 200 || r.mouth.voice != want {
			t.Fatalf("%s -> %s (%d %s)", asked, r.mouth.voice, resp.StatusCode, b)
		}
	}
	for name := range openaiVoices {
		if _, ok := kokoroID[openaiVoices[name]]; !ok {
			t.Fatalf("%s maps to a voice kokoro does not have", name)
		}
	}
	if len(kokoroVoices) != 54 {
		t.Fatalf("Kokoro v1.0 carries 54 voices, not %d", len(kokoroVoices))
	}
}

func TestAnUnknownVoiceNamesTheValidOnes(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.post("/v1/audio/speech", map[string]any{"input": "hi", "voice": "nobody"})
	if resp.StatusCode != 400 {
		t.Fatalf("%d", resp.StatusCode)
	}
	for _, v := range []string{"nobody", "af_heart", "bf_emma", "shimmer"} {
		if !strings.Contains(string(b), v) {
			t.Fatalf("refusal must name %s: %s", v, b)
		}
	}
}

func TestTheLanguageComesFromTheVoice(t *testing.T) {
	for v, want := range map[string]string{"af_heart": "en-us", "bm_george": "en", "ef_dora": "es", "ff_siwis": "fr", "hf_alpha": "hi", "if_sara": "it", "jf_alpha": "ja", "pf_dora": "pt-br", "zm_yunxi": "cmn"} {
		if got := kokoroLang(v); got != want {
			t.Fatalf("%s speaks %s, want %s", v, got, want)
		}
	}
}

func TestSpeedIsHonouredAndBounded(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, _ := r.post("/v1/audio/speech", map[string]any{"input": "hi", "speed": 1.5, "response_format": "wav"})
	if resp.StatusCode != 200 || r.mouth.speed != 1.5 {
		t.Fatalf("speed %v", r.mouth.speed)
	}
	for _, bad := range []float64{0.1, 5} {
		if resp, _ := r.post("/v1/audio/speech", map[string]any{"input": "hi", "speed": bad}); resp.StatusCode != 400 {
			t.Fatalf("speed %v: %d", bad, resp.StatusCode)
		}
	}
	if resp, _ := r.post("/v1/audio/speech", map[string]any{"input": "  "}); resp.StatusCode != 400 {
		t.Fatal("empty input")
	}
	if resp, _ := r.post("/v1/audio/speech", map[string]any{"model": "tts-1", "input": "hi"}); resp.StatusCode != 404 {
		t.Fatal("unknown speech model")
	}
}

// ── the growing transcript ────────────────────────────────────────────────────

func pcmOf(s []float32) []byte { return pcm16(s) }

func settle(t *testing.T, tr *transcript) {
	t.Helper()
	end := time.Now().Add(10 * time.Second)
	for time.Now().Before(end) {
		tr.mu.Lock()
		busy := tr.decoding
		tr.mu.Unlock()
		if !busy {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("decode never settled")
}

func (r *rig) open(body map[string]any) map[string]any {
	r.t.Helper()
	if body == nil {
		body = map[string]any{"model": "whisper", "language": "en"}
	}
	resp, b := r.post("/v1/audio/transcript", body)
	if resp.StatusCode != 201 {
		r.t.Fatalf("open: %d %s", resp.StatusCode, b)
	}
	return decodeJSON(r.t, b)
}

func TestAPushMetersTheAudioItCarried(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	live := r.open(nil)
	resp, b := r.post("/v1/audio/transcript/"+live["id"].(string), pcmOf(tone(7, 0.256)))
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	st := decodeJSON(t, b)
	if st["duration"] != 0.256 || st["seconds"] != 0.256 {
		t.Fatalf("%v", st)
	}
}

func TestASessionBillsEverySecondOnceAndOnlyOnce(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	id := r.open(nil)["id"].(string)
	var billed float64
	for n := range 12 {
		_, b := r.post("/v1/audio/transcript/"+id, pcmOf(tone(n+1, 0.25)))
		billed += decodeJSON(t, b)["duration"].(float64)
	}
	resp, b := r.delete("/v1/audio/transcript/" + id)
	shut := decodeJSON(t, b)
	if resp.StatusCode != 200 || shut["duration"] != 0.0 {
		t.Fatalf("close carries no audio, so it bills none: %d %s", resp.StatusCode, b)
	}
	if math.Abs(billed-3.0) > 1e-9 || math.Abs(shut["seconds"].(float64)-3.0) > 1e-9 {
		t.Fatalf("billed %v, seconds %v; sent 3.0", billed, shut["seconds"])
	}
}

func TestSecondsComeFromTheSamplesNotTheDecoder(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	r.whisper.fail = fmt.Errorf("decoder is unhappy")
	tr := newTranscript(r.sp, "whisper", "en")
	if got := tr.push(quiet(1.5)); got != 1.5 || tr.seconds != 1.5 {
		t.Fatalf("pushed 1.5 s, metered %v", got)
	}
}

// A conversation: three utterances, a second apart.
func conversation() []float32 {
	return cat(tone(1, 2), quiet(1), tone(2, 2), quiet(1), tone(3, 2))
}

func TestSettledSpeechCommitsAndItsAudioIsFreed(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(conversation())
	settle(t, tr)
	if tr.text != "1 2" {
		t.Fatalf("the first two utterances ended a guard-length before the window: %q", tr.text)
	}
	if tr.pending != "3" {
		t.Fatalf("the last is still open: %q", tr.pending)
	}
	// Freed up to the open utterance, less Pad of lead-in for its onset.
	if want := int((6 - Pad) * Rate); len(tr.window) != 8*Rate-want {
		t.Fatalf("window holds %d samples, want %d", len(tr.window), 8*Rate-want)
	}
}

func TestTheOpenTailIsNeverCommittedEarly(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(tone(9, 1.5))
	settle(t, tr)
	if tr.text != "" || tr.pending != "9" {
		t.Fatalf("text %q pending %q", tr.text, tr.pending)
	}
}

func TestCloseCommitsTheRemainder(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(conversation())
	settle(t, tr)
	if err := tr.close(); err != nil {
		t.Fatal(err)
	}
	if tr.text != "1 2 3" || tr.pending != "" {
		t.Fatalf("text %q pending %q", tr.text, tr.pending)
	}
}

func TestTheWindowStaysBoundedUnderContinuousSpeech(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	for v := 1; v <= 30; v++ { // 60 s with no pause at all
		tr.push(tone(v, 2))
		settle(t, tr)
	}
	if len(tr.window) > int((Window+2)*Rate) {
		t.Fatalf("window grew to %.1f s", float64(len(tr.window))/Rate)
	}
	for _, s := range r.whisper.spans {
		if s > Window+2+2*Pad {
			t.Fatalf("a decode covered %.1f s", s)
		}
	}
	if err := tr.close(); err != nil {
		t.Fatal(err)
	}
	var want []string
	for v := 1; v <= 30; v++ {
		want = append(want, strconv.Itoa(v))
	}
	if tr.text != strings.Join(want, " ") {
		t.Fatalf("every tone exactly once, in order: %q", tr.text)
	}
}

func TestTextIsNeverDuplicatedAcrossCuts(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	for v := 1; v <= 10; v++ {
		tr.push(cat(tone(v, 1.5), quiet(0.75)))
		settle(t, tr)
	}
	if err := tr.close(); err != nil {
		t.Fatal(err)
	}
	var want []string
	for v := 1; v <= 10; v++ {
		want = append(want, strconv.Itoa(v))
	}
	if tr.text != strings.Join(want, " ") {
		t.Fatalf("%q", tr.text)
	}
}

func TestAParakeetSessionNamingNoLanguageHearsFirst(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "zh")
	tr := newTranscript(r.sp, "parakeet", "")
	tr.push(tone(4, 1.5)) // under Hearing: not enough to tell a language by
	settle(t, tr)
	if r.parakeet.calls()+r.whisper.calls() != 0 {
		t.Fatal("a session decodes nothing before it has heard which language it is")
	}
	tr.push(tone(4, 1.0))
	settle(t, tr)
	if r.whisper.calls() == 0 || r.parakeet.calls() != 0 {
		t.Fatalf("identified as zh, so whisper: parakeet %d whisper %d", r.parakeet.calls(), r.whisper.calls())
	}
}

func TestTheOpenStatesTheShapeItWants(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	live := r.open(nil)
	if live["rate"] != 16000.0 || live["channels"] != 1.0 || live["format"] != "pcm16" || live["chunk_ms"] != 256.0 {
		t.Fatalf("%v", live)
	}
	if !strings.HasPrefix(live["id"].(string), "atr_") {
		t.Fatal(live["id"])
	}
	// The voice service opens with format/rate/channels and no language.
	resp, b := r.post("/v1/audio/transcript", map[string]any{"model": "parakeet", "format": "pcm16", "rate": 16000, "channels": 1})
	if resp.StatusCode != 201 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	// The ai plane opens with model and an empty language.
	resp, b = r.post("/v1/audio/transcript", map[string]any{"model": "whisper", "language": ""})
	if resp.StatusCode != 201 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

func TestTheTranscriptRefusals(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	if resp, b := r.post("/v1/audio/transcript", map[string]any{"model": "whisper", "rate": 8000}); resp.StatusCode != 400 || !strings.Contains(string(b), "16000") {
		t.Fatalf("a different rate is refused, not resampled: %d %s", resp.StatusCode, b)
	}
	if resp, _ := r.post("/v1/audio/transcript", map[string]any{"model": "not-a-model"}); resp.StatusCode != 404 {
		t.Fatal("unknown model on open")
	}
	if resp, _ := r.post("/v1/audio/transcript/atr_nope", pcmOf(tone(1, 0.25))); resp.StatusCode != 404 {
		t.Fatal("unknown transcript")
	}
	id := r.open(nil)["id"].(string)
	if resp, _ := r.post("/v1/audio/transcript/"+id, pcmOf(tone(1, 3))); resp.StatusCode != 413 {
		t.Fatal("an oversize chunk is refused, not truncated")
	}
	if resp, _ := r.post("/v1/audio/transcript/"+id, []byte{1, 2, 3}); resp.StatusCode != 400 {
		t.Fatal("half a sample shifts every sample after it")
	}
	tr := r.s.live.find(id)
	tr.mu.Lock()
	tr.seconds = Limit
	tr.mu.Unlock()
	if resp, _ := r.post("/v1/audio/transcript/"+id, pcmOf(tone(1, 0.25))); resp.StatusCode != 409 {
		t.Fatal("a full transcript stops taking audio")
	}
	if resp, _ := r.delete("/v1/audio/transcript/" + id); resp.StatusCode != 200 {
		t.Fatal("close")
	}
	if resp, _ := r.delete("/v1/audio/transcript/" + id); resp.StatusCode != 404 {
		t.Fatal("closing twice")
	}
}

func TestAbandonedTranscriptsAreCollected(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	old := newTranscript(r.sp, "whisper", "en")
	r.s.live.begin(old)
	old.mu.Lock()
	old.touched = time.Now().Add(-time.Duration(Idle+1) * time.Second)
	old.mu.Unlock()
	r.s.live.begin(newTranscript(r.sp, "whisper", "en")) // collection happens on the way in
	if r.s.live.find(old.id) != nil {
		t.Fatal("an abandoned session pins its audio forever")
	}
	time.Sleep(20 * time.Millisecond)
	old.mu.Lock()
	freed := old.ended && old.sq.g.(*roomGate).closed
	old.mu.Unlock()
	if !freed {
		t.Fatal("and its detector is freed")
	}
	if old.push(tone(1, 0.25)) != 0 {
		t.Fatal("an ended session takes no audio")
	}
}

func TestSamplesMapInt16OntoTheUnitRange(t *testing.T) {
	got := samples([]byte{0, 0, 0, 0x40, 0, 0x80})
	if got[0] != 0 || got[1] != 0.5 || got[2] != -1 {
		t.Fatal(got)
	}
	back := samples(pcm16([]float32{0, 0.5, -1, 2}))
	if back[0] != 0 || back[1] != 0.5 || back[2] != -32767.0/32768 || back[3] != 32767.0/32768 {
		t.Fatalf("full scale clips, nothing wraps: %v", back)
	}
}

func TestAFailedDecodeSaysSo(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "en")
	r.whisper.fail = fmt.Errorf("decoder is unhappy")
	var logs bytes.Buffer
	restore := captureLogs(&logs)
	defer restore()
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(tone(1, 2))
	settle(t, tr)
	if !strings.Contains(logs.String(), "decoder is unhappy") {
		t.Fatalf("a failed decode must be loud enough to find: %q", logs.String())
	}
	if tr.text != "" || tr.pending != "" {
		t.Fatal("and must not invent text")
	}
}

// ── the squelch: the decode that never runs ───────────────────────────────────

func loud(s float64) []float32 { return tone(30000, s) } // 0.92 of full scale
func hush(s float64) []float32 { return tone(3, s) }     // 0.0001: a room with nobody in it

func TestAQuietRoomCostsNothing(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	var billed float64
	for range 12 {
		billed += tr.push(hush(0.25))
	}
	if billed != 0 || tr.seconds != 0 || len(tr.window) != 0 || r.whisper.calls() != 0 {
		t.Fatalf("billed %v, window %d, decodes %d", billed, len(tr.window), r.whisper.calls())
	}
}

func TestAVoiceIsDecoded(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	var billed float64
	for range 8 {
		billed += tr.push(loud(0.25))
	}
	settle(t, tr)
	if billed != 2 || r.whisper.calls() == 0 {
		t.Fatalf("billed %v decodes %d", billed, r.whisper.calls())
	}
}

func TestTheOnsetThatOpenedTheSquelchIsDecodedWithIt(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	for range 8 {
		if tr.push(hush(0.25)) != 0 {
			t.Fatal("quiet is free")
		}
	}
	opened := tr.push(loud(0.25))
	if opened <= 0.25 {
		t.Fatal("the push that opens carries more than itself")
	}
	if got := len(tr.window); got != int((Lead+0.25)*Rate) {
		t.Fatalf("window %d samples: Lead of lead-in and the push", got)
	}
	if fmt.Sprint(tr.window[:int(Lead*Rate)]) != fmt.Sprint(hush(Lead)) {
		t.Fatal("what precedes the voice is what preceded it")
	}
	if math.Abs(opened-float64(len(tr.window))/Rate) > 1e-9 {
		t.Fatal("all of it billed once")
	}
}

func TestAPauseInsideASentenceDoesNotEndIt(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(loud(0.25))
	gap := []float64{tr.push(hush(0.25)), tr.push(hush(0.25))}
	after := tr.push(loud(0.25))
	if gap[0] != 0.25 || gap[1] != 0.25 || after != 0.25 {
		t.Fatalf("gap %v after %v: half a second of pause is inside the sentence", gap, after)
	}
}

func TestTheSquelchClosesOnceTheRoomStaysQuiet(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	tr.push(loud(0.25))
	var after []float64
	for range 8 {
		after = append(after, tr.push(hush(0.25)))
	}
	var spent float64
	for _, a := range after {
		spent += a
	}
	if spent != 0.75 || spent > Hang {
		t.Fatalf("the hangover and no more: %v", after)
	}
	for _, a := range after[3:] {
		if a != 0 {
			t.Fatalf("quiet from there on is free: %v", after)
		}
	}
}

func TestASessionNeverBillsASecondItDidNotDecode(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	tr := newTranscript(r.sp, "whisper", "en")
	var sent, billed float64
	for _, piece := range [][]float32{hush(1), loud(1), hush(1), loud(1), hush(2)} {
		for at := 0; at < len(piece); at += Chunk / Width {
			p := piece[at:min(at+Chunk/Width, len(piece))]
			billed += tr.push(p)
			sent += float64(len(p)) / Rate
		}
	}
	settle(t, tr)
	if math.Abs(billed-tr.seconds) > 1e-9 || math.Abs(tr.seconds-float64(tr.got)/Rate) > 1e-9 || billed >= sent {
		t.Fatalf("billed %v seconds %v got %v sent %v", billed, tr.seconds, float64(tr.got)/Rate, sent)
	}
}

func TestTheAckReportsZeroForSilence(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	id := r.open(nil)["id"].(string)
	_, b := r.post("/v1/audio/transcript/"+id, pcmOf(hush(0.25)))
	st := decodeJSON(t, b)
	if st["duration"] != 0.0 || st["seconds"] != 0.0 {
		t.Fatal(st)
	}
	// A push carrying no audio is a no-op, which is how a client reads the
	// newest text without sending more.
	resp, b := r.post("/v1/audio/transcript/"+id, []byte{})
	if resp.StatusCode != 200 || decodeJSON(t, b)["duration"] != 0.0 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
}

// ── which process owns the session ────────────────────────────────────────────

func TestOpenNamesWhereItLives(t *testing.T) {
	t.Setenv("POD_IP", "10.244.3.17")
	if got := here("8000"); got != "http://10.244.3.17:8000" {
		t.Fatal(got)
	}
	t.Setenv("POD_IP", "")
	if here("8000") != "" {
		t.Fatal("unset is empty, not a guess")
	}
	r := newRig(t, alwaysVAD{}, "en")
	r.s.here = "http://10.244.3.17:8000"
	if r.open(nil)["at"] != "http://10.244.3.17:8000" {
		t.Fatal("the field is on the open response")
	}
}

// ── fairness: every session gets a turn ───────────────────────────────────────

// queuedEar serializes its decodes through a real worker, as the recognizers
// do, and counts them per session by the tone each session speaks in.
type queuedEar struct {
	run  worker
	mu   sync.Mutex
	seen map[int]int
}

func (e *queuedEar) hear(pcm []float32, lang string) (heard, error) {
	e.run.do(func() {
		v := 0
		for _, x := range pcm {
			if x != 0 {
				v = int(math.Round(float64(x) * 32768))
				break
			}
		}
		e.mu.Lock()
		e.seen[v]++
		e.mu.Unlock()
		time.Sleep(30 * time.Millisecond)
	})
	return heard{}, nil
}

func TestNoSessionIsStarvedByAnother(t *testing.T) {
	q := &queuedEar{run: newWorker(), seen: map[int]int{}}
	sp := &speech{ears: map[string]ear{"whisper": q}, vad: alwaysVAD{}, lid: fixedLID{"en"}}
	var sessions []*transcript
	for range 3 {
		sessions = append(sessions, newTranscript(sp, "whisper", "en"))
	}
	stop := time.Now().Add(2 * time.Second)
	var wg sync.WaitGroup
	for i, tr := range sessions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(stop) {
				tr.push(tone(7+i, 1.0))
				time.Sleep(20 * time.Millisecond)
			}
		}()
	}
	wg.Wait()
	q.mu.Lock()
	seen := map[int]int{}
	for k, v := range q.seen {
		seen[k] = v
	}
	q.mu.Unlock()
	for i := range sessions {
		if seen[7+i] == 0 {
			t.Fatalf("session %d ran zero decodes while the others ran %v", i, seen)
		}
	}
	for _, tr := range sessions {
		tr.mu.Lock()
		tr.closed = true
		tr.mu.Unlock()
	}
}

// ── health and weights ────────────────────────────────────────────────────────

func TestHealthIsReadyOnlyOnceModelsLoad(t *testing.T) {
	s := &server{}
	srv := httptest.NewServer(s.routes())
	defer srv.Close()
	resp, _ := http.Get(srv.URL + "/healthz")
	if resp.StatusCode != 503 {
		t.Fatalf("loading: %d", resp.StatusCode)
	}
	resp, _ = http.Get(srv.URL + "/v1/models")
	if resp.StatusCode != 503 {
		t.Fatalf("a model route while loading: %d", resp.StatusCode)
	}
	s.sp.Store(&speech{ears: map[string]ear{"parakeet": &namingEar{}, "whisper": &namingEar{}}, mouths: map[string]mouth{"kokoro": &toneMouth{}}})
	resp, _ = http.Get(srv.URL + "/healthz")
	if resp.StatusCode != 200 {
		t.Fatalf("ready: %d", resp.StatusCode)
	}
	resp, _ = http.Get(srv.URL + "/v1/models")
	b, _ := io.ReadAll(resp.Body)
	for _, m := range []string{"parakeet", "whisper", "kokoro"} {
		if !strings.Contains(string(b), `"`+m+`"`) {
			t.Fatalf("/v1/models must list %s: %s", m, b)
		}
	}
}

func TestTheManifestNamesEveryModelFile(t *testing.T) {
	all, err := manifest()
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, w := range all {
		have[w.path] = true
	}
	for _, p := range []string{
		"parakeet/encoder.int8.onnx", "parakeet/decoder.int8.onnx", "parakeet/joiner.int8.onnx", "parakeet/tokens.txt",
		"whisper/encoder.int8.onnx", "whisper/decoder.int8.onnx", "whisper/tokens.txt",
		"lid/encoder.int8.onnx", "lid/decoder.int8.onnx",
		"kokoro/model.onnx", "kokoro/voices.bin", "kokoro/tokens.txt", "kokoro/lexicon-zh.txt",
		"vad/silero_vad.onnx",
	} {
		if !have[p] {
			t.Fatalf("weights.sum does not name %s", p)
		}
	}
}

func TestEscapeIsSigV4s(t *testing.T) {
	if got := escape("speech/kokoro/espeak-ng-data/voices/!v/Mr serious"); got != "speech/kokoro/espeak-ng-data/voices/%21v/Mr%20serious" {
		t.Fatal(got)
	}
}

func TestFetchVerifiesWhatItDownloads(t *testing.T) {
	good := []byte("the weights")
	sum := sha256.Sum256(good)
	served := good
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		if r.URL.EscapedPath() != "/models/speech/vad/silero_vad.onnx" {
			http.NotFound(w, r)
			return
		}
		w.Write(served)
	}))
	defer srv.Close()
	saved := weightsSum
	defer func() { weightsSum = saved }()
	weightsSum = fmt.Sprintf("%s %d vad/silero_vad.onnx\n", hex.EncodeToString(sum[:]), len(good))
	st := store{endpoint: srv.URL, bucket: "models", region: "us-east-1", key: "AKID", secret: "SECRET"}

	dir := t.TempDir()
	if err := fetch(context.Background(), dir, st); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "vad/silero_vad.onnx")); !bytes.Equal(b, good) {
		t.Fatal("the file is not what was served")
	}
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKID/") || !strings.Contains(auth, "/us-east-1/s3/aws4_request") {
		t.Fatalf("unsigned: %q", auth)
	}

	// A corrupted download never lands under its name.
	served = []byte("the weighTs")
	dir = t.TempDir()
	if err := fetch(context.Background(), dir, st); err == nil {
		t.Fatal("a download that does not hash to weights.sum must fail")
	}
	if fileExists(filepath.Join(dir, "vad/silero_vad.onnx")) || fileExists(filepath.Join(dir, "vad/silero_vad.onnx.part")) {
		t.Fatal("and must leave nothing behind")
	}

	// Missing with nowhere to fetch from: the service must not start without its weights.
	if err := fetch(context.Background(), t.TempDir(), store{}); err == nil {
		t.Fatal("missing weights with no store to fetch from is an error, not a silent start")
	}
}

func captureLogs(w io.Writer) func() {
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(w, nil)))
	return func() { slog.SetDefault(prev) }
}

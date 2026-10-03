package main

// The end-to-end gate: real weights, real audio, real decodes.
//
// speech_test.go replaces every model, because the arithmetic is what it is
// about and a stand-in makes it exact. The cost is real: a suite where the
// recognizer is only ever a stand-in cannot tell a working decoder from a
// deleted one. Nothing here is replaced. kokoro speaks, and parakeet and whisper
// have to say it back — so a model that is not loaded, audio handed over at the
// wrong rate or scale, or a container ffmpeg cannot read all fail here.
//
// It runs where the weights are: the models directory the service itself reads
// (MODELS). Without them it skips, and says which file it looked for.

import (
	"context"
	"encoding/json"
	"math"
	"math/rand"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"
)

const said = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

var (
	liveOnce sync.Once
	liveSp   *speech
	liveErr  error
	liveSkip string
)

func live(t *testing.T) *speech {
	t.Helper()
	liveOnce.Do(func() {
		dir := envOr("MODELS", "/models")
		all, err := manifest()
		if err != nil {
			liveErr = err
			return
		}
		for _, w := range all {
			if fi, err := os.Stat(filepath.Join(dir, w.path)); err != nil || fi.Size() != w.size {
				liveSkip = "no weights under " + dir + " (looked for " + w.path + "); set MODELS to the models directory"
				return
			}
		}
		liveSp, liveErr = load(dir)
	})
	if liveSkip != "" {
		t.Skip(liveSkip)
	}
	if liveErr != nil {
		t.Fatal(liveErr)
	}
	return liveSp
}

// speak has kokoro say text, at its own 24 kHz.
func speak(t *testing.T, sp *speech, text, voice string) []float32 {
	t.Helper()
	s, err := sp.mouths["kokoro"].speak(text, voice, 1)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// at16k resamples kokoro's 24 kHz to what a recognizer hears, through ffmpeg —
// the same step a browser takes before it pushes raw audio.
func at16k(t *testing.T, s []float32) []float32 {
	t.Helper()
	pcm, err := decode(context.Background(), wav(s, Spoken), 0)
	if err != nil {
		t.Fatal(err)
	}
	return pcm
}

// wer is the word error rate of got against want, on lowercased words with the
// punctuation stripped.
func wer(want, got string) float64 {
	norm := func(s string) []string {
		return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\'' })
	}
	a, b := norm(want), norm(got)
	d := make([]int, len(b)+1)
	for j := range d {
		d[j] = j
	}
	for i := 1; i <= len(a); i++ {
		prev := d[0]
		d[0] = i
		for j := 1; j <= len(b); j++ {
			cur := d[j]
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			d[j] = min(d[j]+1, d[j-1]+1, prev+cost)
			prev = cur
		}
	}
	return float64(d[len(b)]) / float64(len(a))
}

func TestLiveKokoroSpeaksAndBothRecognizersSayItBack(t *testing.T) {
	sp := live(t)
	r := &rig{t: t, s: &server{maxIn: 100 << 20}}
	r.s.sp.Store(sp)
	r.srv = httptest.NewServer(r.s.routes())
	defer r.srv.Close()

	resp, mp3 := r.post("/v1/audio/speech", map[string]any{"model": "kokoro", "input": said, "voice": "af_heart"})
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "audio/mpeg" {
		t.Fatalf("speech: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	for _, model := range []string{"parakeet", "whisper"} {
		resp, b := r.transcribe(mp3, map[string][]string{"model": {model}, "response_format": {"verbose_json"}, "timestamp_granularities[]": {"word", "segment"}})
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", model, resp.StatusCode, b)
		}
		var body struct {
			Text     string  `json:"text"`
			Duration float64 `json:"duration"`
			Words    []word  `json:"words"`
		}
		json.Unmarshal(b, &body)
		t.Logf("%s heard %q (%.2f s)", model, body.Text, body.Duration)
		if e := wer(said, body.Text); e != 0 {
			t.Fatalf("%s: word error %.3f on a clean sentence: %q", model, e, body.Text)
		}
		if body.Duration < 4 {
			t.Fatalf("%s: duration %.2f is not the audio's length", model, body.Duration)
		}
		// Every word timed, in order, inside the audio.
		if len(body.Words) != len(strings.Fields(said)) {
			t.Fatalf("%s: %d words timed for %d said: %v", model, len(body.Words), len(strings.Fields(said)), body.Words)
		}
		for i, w := range body.Words {
			if w.End < w.Start || w.End > body.Duration+0.05 || (i > 0 && w.Start < body.Words[i-1].Start) {
				t.Fatalf("%s: word %d timed %+v", model, i, w)
			}
		}
	}
}

// Every container a browser records: Chrome and Firefox webm/opus, Firefox
// ogg/opus, Safari mp4/aac, and the two files people upload.
func TestLiveEveryBrowserContainerTranscribes(t *testing.T) {
	sp := live(t)
	src := wav(speak(t, sp, said, "af_heart"), Spoken)
	dir := t.TempDir()
	in := filepath.Join(dir, "said.wav")
	os.WriteFile(in, src, 0o644)
	for name, args := range map[string][]string{
		"webm": {"-c:a", "libopus", "-f", "webm"},
		"ogg":  {"-c:a", "libopus", "-f", "ogg"},
		"m4a":  {"-c:a", "aac", "-f", "mp4"},
		"mp3":  {"-c:a", "libmp3lame", "-f", "mp3"},
		"wav":  {"-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-f", "wav"},
	} {
		out := filepath.Join(dir, "recorded."+name)
		cmd := exec.Command("ffmpeg", append(append([]string{"-nostdin", "-loglevel", "error", "-y", "-i", in}, args...), out)...)
		if b, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%s: %v %s", name, err, b)
		}
		data, _ := os.ReadFile(out)
		pcm, err := decode(context.Background(), data, 0)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		got, err := sp.transcribe("parakeet", "", pcm)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %q", name, got.text)
		if e := wer(said, got.text); e != 0 {
			t.Fatalf("%s: word error %.3f: %q", name, e, got.text)
		}
	}
}

// shared is the share of want's Han characters that appear in got.
func shared(want, got string) float64 {
	have := map[rune]int{}
	for _, r := range got {
		have[r]++
	}
	n, hit := 0, 0
	for _, r := range want {
		if !unicode.Is(unicode.Han, r) {
			continue
		}
		n++
		if have[r] > 0 {
			have[r]--
			hit++
		}
	}
	return float64(hit) / float64(n)
}

func hasScript(s string, tab *unicode.RangeTable) bool {
	for _, r := range s {
		if unicode.Is(tab, r) {
			return true
		}
	}
	return false
}

// A language parakeet lacks, named or not, is whisper's.
func TestLiveParakeetHandsWhatItLacksToWhisper(t *testing.T) {
	sp := live(t)
	for _, c := range []struct {
		voice, text, lang string
		script            *unicode.RangeTable
	}{
		{"zf_xiaoxiao", "今天天气很好，我们一起去公园散步吧。", "zh", unicode.Han},
		{"hf_alpha", "नमस्ते, आज मौसम बहुत अच्छा है और हम पार्क जा रहे हैं।", "hi", unicode.Devanagari},
	} {
		pcm := at16k(t, speak(t, sp, c.text, c.voice))
		_, engine, _, err := sp.route("parakeet", "", func() []float32 { return spoken(pcm, sp.vad.regions(pcm)) })
		if err != nil {
			t.Fatal(err)
		}
		if engine != "whisper" {
			t.Fatalf("%s: unnamed %s audio routed to %s", c.lang, c.lang, engine)
		}
		got, err := sp.transcribe("parakeet", "", pcm)
		if err != nil {
			t.Fatal(err)
		}
		named, err := sp.transcribe("parakeet", c.lang, pcm)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s identified: %q  named: %q", c.lang, got.text, named.text)
		if !hasScript(got.text, c.script) || !hasScript(named.text, c.script) {
			t.Fatalf("%s: whisper did not answer in %s", c.lang, c.lang)
		}
		if c.lang == "zh" {
			// Characters, not words: Chinese is written without spaces.
			if share := shared(c.text, got.text); share < 0.8 {
				t.Fatalf("zh: only %.0f%% of the characters said came back: %q", share*100, got.text)
			}
		}
	}
	// And a European language stays with parakeet.
	pcm := at16k(t, speak(t, sp, "Buenos días, hoy vamos a la playa con mis amigos.", "ef_dora"))
	_, engine, _, _ := sp.route("parakeet", "", func() []float32 { return spoken(pcm, sp.vad.regions(pcm)) })
	if engine != "parakeet" {
		t.Fatalf("Spanish routed to %s", engine)
	}
}

// pushAll feeds audio through the growing transcript in browser-sized chunks.
func pushAll(tr *transcript, pcm []float32) (billed float64) {
	step := Chunk / Width
	for at := 0; at < len(pcm); at += step {
		billed += tr.push(pcm[at:min(at+step, len(pcm))])
	}
	return billed
}

func TestLiveAGrowingTranscriptHearsTheSentence(t *testing.T) {
	sp := live(t)
	pcm := at16k(t, speak(t, sp, said, "af_heart"))
	for _, model := range []string{"parakeet", "whisper"} {
		tr := newTranscript(sp, model, "en")
		billed := pushAll(tr, pcm)
		if err := tr.close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %q billed %.2f s of %.2f", model, tr.text, billed, float64(len(pcm))/Rate)
		if e := wer(said, tr.text); e != 0 {
			t.Fatalf("%s: word error %.3f: %q", model, e, tr.text)
		}
		if billed > float64(len(pcm))/Rate+1e-9 || math.Abs(billed-tr.seconds) > 1e-9 {
			t.Fatalf("%s: billed %.3f for %.3f s submitted", model, billed, float64(len(pcm))/Rate)
		}
	}
}

func TestLiveItDecodesWhileTheTranscriptIsStillOpen(t *testing.T) {
	sp := live(t)
	pcm := at16k(t, speak(t, sp, said, "af_heart"))
	tr := newTranscript(sp, "parakeet", "en")
	pushAll(tr, pcm[:len(pcm)/2])
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		tr.mu.Lock()
		busy, heard := tr.decoding, tr.text+" "+tr.pending
		tr.mu.Unlock()
		if !busy && strings.TrimSpace(heard) != "" {
			t.Logf("mid-stream: %q", heard)
			if !strings.Contains(strings.ToLower(heard), "fox") {
				t.Fatalf("nothing right decoded mid-stream: %q", heard)
			}
			tr.close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("no words before the end")
}

// roomTone is a fan, a laptop, an open microphone in an empty room: noise at a
// level a meeting really carries, not digital silence a detector cannot fail.
func roomTone(seconds, dbfs float64) []float32 {
	rng := rand.New(rand.NewSource(5))
	n := int(seconds * Rate)
	raw := make([]float64, n)
	for i := range raw {
		raw[i] = rng.NormFloat64()
	}
	out := make([]float32, n)
	peak := 0.0
	warm := make([]float64, n)
	for i := range raw { // a 16-tap moving average: warm noise, like a fan
		s := 0.0
		for k := max(0, i-15); k <= i; k++ {
			s += raw[k]
		}
		warm[i] = s / 16
		peak = math.Max(peak, math.Abs(warm[i]))
	}
	for i := range out {
		out[i] = float32(warm[i] / peak * math.Pow(10, dbfs/20))
	}
	return out
}

func TestLiveAnEmptyRoomIsNeverDecoded(t *testing.T) {
	sp := live(t)
	tr := newTranscript(sp, "parakeet", "en")
	defer tr.close()
	if billed := pushAll(tr, roomTone(6, -45)); billed != 0 || tr.got != 0 {
		t.Fatalf("a room with nobody in it billed %.2f s", billed)
	}
	if billed := pushAll(tr, at16k(t, speak(t, sp, said, "af_heart"))); billed <= 0 {
		t.Fatal("and a voice in the same room is heard")
	}
}

func TestLiveTheWordAfterAPauseIsStillThere(t *testing.T) {
	sp := live(t)
	turns := []string{"Right, so the migration landed yesterday and the error rate is flat.", "Kubernetes rescheduled the workers twice overnight."}
	meeting := cat(roomTone(2.5, -45), at16k(t, speak(t, sp, turns[0], "am_michael")), roomTone(0.5, -45),
		at16k(t, speak(t, sp, turns[1], "bf_emma")), roomTone(2.5, -45))
	tr := newTranscript(sp, "parakeet", "en")
	billed := pushAll(tr, meeting)
	if err := tr.close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("meeting: %q billed %.2f of %.2f s", tr.text, billed, float64(len(meeting))/Rate)
	if e := wer(strings.Join(turns, " "), tr.text); e > 0.1 {
		t.Fatalf("the squelch lost words: error %.3f, heard %q", e, tr.text)
	}
	if billed >= float64(len(meeting))/Rate {
		t.Fatal("the lulls were charged for")
	}
}

// The realtime factor of each model on this machine at the pod's width: wall
// seconds per second of audio, on the clean sentence, after a warm-up. A model
// above 1.0 cannot keep up with a speaker on this many cores.
func TestLiveRealtimeFactor(t *testing.T) {
	sp := live(t)
	spoke := speak(t, sp, said+" "+said, "af_heart")
	pcm := at16k(t, spoke)
	audio := float64(len(pcm)) / Rate
	for _, m := range []string{"parakeet", "whisper"} {
		sp.ears[m].hear(pcm, "en") // warm
		began := time.Now()
		sp.ears[m].hear(pcm, "en")
		t.Logf("%-8s RTF %.3f (%.2f s audio, %d threads)", m, time.Since(began).Seconds()/audio, audio, runtime.GOMAXPROCS(0))
	}
	began := time.Now()
	out, _ := sp.mouths["kokoro"].speak(said+" "+said, "af_heart", 1)
	t.Logf("%-8s RTF %.3f (%.2f s audio)", "kokoro", time.Since(began).Seconds()/(float64(len(out))/Spoken), float64(len(out))/Spoken)
	began = time.Now()
	sp.lid.identify(pcm)
	t.Logf("%-8s %.3f s per identification", "lid", time.Since(began).Seconds())
	if b, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(l, "VmRSS") || strings.HasPrefix(l, "VmHWM") {
				t.Log(strings.Join(strings.Fields(l), " "))
			}
		}
	}
}

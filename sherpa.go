package main

// The native half: every call into sherpa-onnx lives in this file, and nothing
// outside it knows the runtime exists. The rest of the service talks to the
// interfaces in speech.go, which is what lets its arithmetic be tested against
// stand-ins without a weight on disk.
//
// sherpa-onnx is C++ over ONNX Runtime. Two of its failure modes end the
// process rather than return an error — an unknown whisper language and a VAD
// window of the wrong size both call exit(-1) — so this file validates before it
// calls, and never hands the runtime an empty buffer (taking &samples[0] of one
// is a Go panic before it is a C bug).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unicode"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-linux"
)

// worker runs one model's calls one at a time, in the order they arrived.
//
// One at a time because each call already spreads across every core the pod
// has (NumThreads below), so a second concurrent call buys a longer queue, not
// more throughput. In arrival order because the queue IS the fairness: a pass
// over one transcript waits behind the passes already queued, never ahead of
// them. An unbuffered channel queues its senders FIFO, so there is no second
// structure to keep in step.
type worker chan func()

func newWorker() worker {
	w := make(worker)
	go func() {
		for f := range w {
			f()
		}
	}()
	return w
}

func (w worker) do(f func()) {
	done := make(chan struct{})
	w <- func() { f(); close(done) }
	<-done
}

// threads is how many cores one call may spread across: the pod's CPU limit,
// which GOMAXPROCS already reads from the cgroup.
func threads() int { return runtime.GOMAXPROCS(0) }

// shortest is the least audio handed to a recognizer. A transducer subsamples
// its input eightfold and whisper wants frames to attend over; a fragment
// shorter than this is padded with silence, which changes no words.
const shortest = Rate

func padded(pcm []float32) []float32 {
	if len(pcm) >= shortest {
		return pcm
	}
	out := make([]float32, shortest)
	copy(out, pcm)
	return out
}

// recognizer is a sherpa offline recognizer behind a worker.
type recognizer struct {
	rec     *sherpa.OfflineRecognizer
	cfg     sherpa.OfflineRecognizerConfig
	whisper bool
	run     worker
}

func must(path string) (string, error) {
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("weights: %w", err)
	}
	return path, nil
}

func files(dir string, names ...string) ([]string, error) {
	out := make([]string, len(names))
	for i, n := range names {
		p, err := must(filepath.Join(dir, n))
		if err != nil {
			return nil, err
		}
		out[i] = p
	}
	return out, nil
}

// newParakeet loads NVIDIA Parakeet TDT 0.6B v3 (int8): a transducer, so its
// tokens carry their own start and duration and words are timed for free.
func newParakeet(dir string) (*recognizer, error) {
	f, err := files(filepath.Join(dir, "parakeet"), "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
	if err != nil {
		return nil, err
	}
	var c sherpa.OfflineRecognizerConfig
	c.FeatConfig = sherpa.FeatureConfig{SampleRate: Rate, FeatureDim: 80}
	c.ModelConfig.Transducer = sherpa.OfflineTransducerModelConfig{Encoder: f[0], Decoder: f[1], Joiner: f[2]}
	c.ModelConfig.Tokens = f[3]
	c.ModelConfig.ModelType = "nemo_transducer"
	c.ModelConfig.NumThreads = threads()
	c.ModelConfig.Provider = "cpu"
	c.DecodingMethod = "greedy_search"
	return open(c, false)
}

// newWhisper loads Whisper large-v3-turbo (int8). The decoder is our own export
// with the alignment heads' cross-attention as a fourth output, which is what
// lets sherpa time each token by DTW; the published export has three outputs and
// times nothing.
func newWhisper(dir string) (*recognizer, error) {
	f, err := files(filepath.Join(dir, "whisper"), "encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")
	if err != nil {
		return nil, err
	}
	var c sherpa.OfflineRecognizerConfig
	c.FeatConfig = sherpa.FeatureConfig{SampleRate: Rate, FeatureDim: 128}
	c.ModelConfig.Whisper = sherpa.OfflineWhisperModelConfig{
		Encoder: f[0], Decoder: f[1], Task: "transcribe",
		TailPaddings: -1, EnableTokenTimestamps: 1,
	}
	c.ModelConfig.Tokens = f[2]
	c.ModelConfig.ModelType = "whisper"
	c.ModelConfig.NumThreads = threads()
	c.ModelConfig.Provider = "cpu"
	c.DecodingMethod = "greedy_search"
	return open(c, true)
}

func open(c sherpa.OfflineRecognizerConfig, whisper bool) (*recognizer, error) {
	rec := sherpa.NewOfflineRecognizer(&c)
	if rec == nil {
		return nil, fmt.Errorf("sherpa refused the recognizer config")
	}
	return &recognizer{rec: rec, cfg: c, whisper: whisper, run: newWorker()}, nil
}

// hear decodes one span. lang matters only to whisper: its language is part of
// the recognizer's config, not the stream's (v1.13.8 reads no per-stream option
// for whisper), so it is set and the span decoded inside the same worker turn —
// nothing else can run between the two. lang must already be a code whisper
// knows (language.go); an unknown one makes sherpa exit the process.
func (r *recognizer) hear(pcm []float32, lang string) (heard, error) {
	var out heard
	r.run.do(func() {
		if r.whisper && r.cfg.ModelConfig.Whisper.Language != lang {
			r.cfg.ModelConfig.Whisper.Language = lang
			r.rec.SetConfig(&r.cfg)
		}
		s := sherpa.NewOfflineStream(r.rec)
		defer sherpa.DeleteOfflineStream(s)
		s.AcceptWaveform(Rate, padded(pcm))
		r.rec.Decode(s)
		res := s.GetResult()
		out = heard{text: strings.TrimSpace(res.Text), words: words(res.Tokens, res.Timestamps, res.Durations, r.whisper)}
	})
	return out, nil
}

// words groups tokens into timed words.
//
// A word starts where a token starts with a space (whisper's BPE) or "▁"
// (Parakeet's SentencePiece). Scripts written without spaces — Han, kana, Thai
// and their neighbours — have no such marker, so each of their tokens is a word
// of its own, which is how whisper's own word timing splits them.
//
// A token ends where its duration says, or where the next one begins when the
// model gives no duration. No timestamps at all means the model timed nothing,
// and the answer is no words rather than invented ones.
func words(tokens []string, at, dur []float32, whisper bool) []word {
	if len(tokens) == 0 || len(at) != len(tokens) {
		return nil
	}
	end := func(i int) float64 {
		if len(dur) == len(tokens) && dur[i] > 0 {
			return float64(at[i] + dur[i])
		}
		if i+1 < len(at) {
			return float64(at[i+1])
		}
		return float64(at[i])
	}
	var out []word
	for i, t := range tokens {
		if whisper && strings.HasPrefix(t, "<|") {
			continue // a special token: language, task, timestamp
		}
		bare := strings.TrimLeft(t, " ▁")
		if bare == "" {
			continue
		}
		fresh := len(out) == 0 || strings.HasPrefix(t, " ") || strings.HasPrefix(t, "▁") || unspaced(bare) || unspaced(out[len(out)-1].Word)
		if fresh {
			out = append(out, word{Word: bare, Start: float64(at[i]), End: end(i)})
			continue
		}
		w := &out[len(out)-1]
		w.Word += bare
		w.End = end(i)
	}
	return out
}

// unspaced reports whether s is written in a script that does not put spaces
// between words.
func unspaced(s string) bool {
	for _, r := range s {
		if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai, unicode.Lao, unicode.Myanmar, unicode.Khmer) {
			return true
		}
	}
	return false
}

// identifier is spoken language identification on a small whisper (tiny,
// int8): one encoder pass and one decoder step, a fraction of a decode.
type identifier struct {
	lid *sherpa.SpokenLanguageIdentification
	run worker
}

func newIdentifier(dir string) (*identifier, error) {
	f, err := files(filepath.Join(dir, "lid"), "encoder.int8.onnx", "decoder.int8.onnx")
	if err != nil {
		return nil, err
	}
	c := sherpa.SpokenLanguageIdentificationConfig{
		Whisper:    sherpa.SpokenLanguageIdentificationWhisperConfig{Encoder: f[0], Decoder: f[1], TailPaddings: -1},
		NumThreads: threads(),
		Provider:   "cpu",
	}
	lid := sherpa.NewSpokenLanguageIdentification(&c)
	if lid == nil {
		return nil, fmt.Errorf("sherpa refused the language identifier config")
	}
	return &identifier{lid: lid, run: newWorker()}, nil
}

func (l *identifier) identify(pcm []float32) (string, error) {
	var lang string
	l.run.do(func() {
		s := l.lid.CreateStream()
		defer sherpa.DeleteOfflineStream(s)
		s.AcceptWaveform(Rate, padded(pcm))
		lang = l.lid.Compute(s).Lang
	})
	if lang == "" {
		return "", fmt.Errorf("language identification returned nothing")
	}
	return lang, nil
}

// kokoro is Kokoro-82M multi-lang v1.0: 54 voices, one model, the language of
// each request taken from its voice and passed per call.
type kokoro struct {
	tts *sherpa.OfflineTts
	run worker
}

func newKokoro(dir string) (*kokoro, error) {
	f, err := files(filepath.Join(dir, "kokoro"), "model.onnx", "voices.bin", "tokens.txt", "lexicon-zh.txt", "espeak-ng-data")
	if err != nil {
		return nil, err
	}
	// Han characters are read through the Chinese lexicon; everything else goes
	// through espeak-ng in the language the request passes (sherpa 1.13.8 takes
	// that path whenever a language is set, and the model's own default is
	// en-us, so it always is). That is why only the Chinese lexicon ships.
	var c sherpa.OfflineTtsConfig
	c.Model.Kokoro = sherpa.OfflineTtsKokoroModelConfig{
		Model: f[0], Voices: f[1], Tokens: f[2], Lexicon: f[3], DataDir: f[4], LengthScale: 1.0,
	}
	c.Model.NumThreads = threads()
	c.Model.Provider = "cpu"
	c.MaxNumSentences = 1
	c.SilenceScale = 0.2
	t := sherpa.NewOfflineTts(&c)
	if t == nil {
		return nil, fmt.Errorf("sherpa refused the kokoro config")
	}
	if n := t.NumSpeakers(); n != len(kokoroVoices) {
		return nil, fmt.Errorf("kokoro has %d voices; this build names %d", n, len(kokoroVoices))
	}
	if r := t.SampleRate(); r != Spoken {
		return nil, fmt.Errorf("kokoro speaks at %d Hz; this build expects %d", r, Spoken)
	}
	return &kokoro{tts: t, run: newWorker()}, nil
}

func (k *kokoro) speak(text, voice string, speed float32) ([]float32, error) {
	sid, ok := kokoroID[voice]
	if !ok {
		return nil, fmt.Errorf("kokoro has no voice %q", voice)
	}
	extra, _ := json.Marshal(map[string]string{"lang": kokoroLang(voice)})
	var out *sherpa.GeneratedAudio
	k.run.do(func() {
		out = k.tts.GenerateWithConfig(text, &sherpa.GenerationConfig{Sid: sid, Speed: speed, SilenceScale: 0.2, Extra: extra}, nil)
	})
	if out == nil || len(out.Samples) == 0 {
		return nil, fmt.Errorf("kokoro produced no audio for this input")
	}
	return out.Samples, nil
}

// silero is the voice activity detector, used two ways: as a session's squelch
// (one detector per session, fed push by push, so it remembers the room) and as
// a segmenter that cuts a span of audio at its pauses.
type silero struct{ path string }

func newSilero(dir string) (*silero, error) {
	p, err := must(filepath.Join(dir, "vad", "silero_vad.onnx"))
	if err != nil {
		return nil, err
	}
	return &silero{path: p}, nil
}

// window is silero's own: 512 samples, 32 ms at 16 kHz. Not a choice.
const window = 512

func (s *silero) detector(silence, longest float32, buffer float32) *sherpa.VoiceActivityDetector {
	c := sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model: s.path, Threshold: Open, MinSilenceDuration: silence,
			MinSpeechDuration: 0, WindowSize: window, MaxSpeechDuration: longest,
		},
		SampleRate: Rate, NumThreads: 1, Provider: "cpu",
	}
	return sherpa.NewVoiceActivityDetector(&c, buffer)
}

// gate is one session's squelch detector. silero's own state machine is the
// squelch's: it opens on a window above Open (0.5), holds through anything above
// Open-0.15 (0.35, the same distance apart as the thresholds it replaced), and
// closes after Hang of quiet. MaxSpeechDuration is the session's whole limit so
// it never switches to its split-forcing thresholds mid-sentence.
type sileroGate struct{ vad *sherpa.VoiceActivityDetector }

func (s *silero) gate() gate {
	return &sileroGate{vad: s.detector(Hang, Limit+1, 30)}
}

// voiced feeds one push and reports whether the squelch was open during any
// window of it — the same "any window" rule the detector applies inside one call.
// A push that is not a whole number of windows leaves its tail in the detector,
// which scores it with the next push rather than padding it with silence that
// was never in the room. A 256 ms push at this rate is exactly eight windows.
func (g *sileroGate) voiced(pcm []float32) bool {
	if len(pcm) == 0 {
		return false
	}
	g.vad.AcceptWaveform(pcm)
	open := g.vad.IsSpeech()
	g.vad.Clear() // closed utterances queue inside the detector; nothing reads them here
	return open
}

func (g *sileroGate) close() { sherpa.DeleteVoiceActivityDetector(g.vad) }

// regions cuts audio at its pauses: the spans silero calls speech, in samples.
// A pause is Pause of quiet. A span longer than Longest is split by silero's own
// rule (it raises its threshold to force a cut) and, failing that, by length in
// regions' caller, because whisper decodes at most thirty seconds at a time.
func (s *silero) regions(pcm []float32) []span {
	if len(pcm) == 0 {
		return nil
	}
	v := s.detector(float32(Pause), float32(Longest), 60)
	defer sherpa.DeleteVoiceActivityDetector(v)
	var out []span
	drain := func() {
		for !v.IsEmpty() {
			f := v.Front()
			out = append(out, span{f.Start, f.Start + len(f.Samples)})
			v.Pop()
		}
	}
	// One window per call: the detector stamps a stretch's start from where the
	// call that opened it ENDED, so a longer call puts every start late by up to
	// its own length — a second-long call cut "The quick" off a sentence.
	for at := 0; at < len(pcm); at += window {
		v.AcceptWaveform(pcm[at:min(at+window, len(pcm))])
		drain()
	}
	v.Flush()
	drain()
	return out
}

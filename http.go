package main

// One OpenAI-shaped surface. Nothing here authenticates: the ai plane
// (hanzoai/ai, behind api.hanzo.ai) fronts this service, authenticates every
// caller and meters what it is told — `duration` on a transcription, characters
// on speech, `seconds` on a growing transcript. A provider row pointing at
// http://speech.hanzo.svc/v1 is the whole integration.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type server struct {
	here  string // where THIS process answers, for a transcript's `at`
	sp    atomic.Pointer[speech]
	live  sessions
	maxIn int64 // largest upload accepted
}

func (s *server) routes() http.Handler {
	m := http.NewServeMux()
	m.HandleFunc("GET /healthz", s.healthz)
	m.HandleFunc("GET /v1/models", s.ready(s.models))
	m.HandleFunc("POST /v1/audio/transcriptions", s.ready(s.transcriptions))
	m.HandleFunc("POST /v1/audio/speech", s.ready(s.speech))
	m.HandleFunc("POST /v1/audio/transcript", s.ready(s.transcriptOpen))
	m.HandleFunc("POST /v1/audio/transcript/{id}", s.ready(s.transcriptPush))
	m.HandleFunc("DELETE /v1/audio/transcript/{id}", s.ready(s.transcriptClose))
	m.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fail(w, http.StatusNotFound, "no route %s %s", r.Method, r.URL.Path)
	})
	return m
}

// healthz is ready only once every model has loaded: a pod that answers before
// then would take traffic it can only refuse.
func (s *server) healthz(w http.ResponseWriter, r *http.Request) {
	if s.sp.Load() == nil {
		reply(w, http.StatusServiceUnavailable, map[string]any{"ok": false, "loading": true})
		return
	}
	reply(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *server) ready(h func(http.ResponseWriter, *http.Request, *speech)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sp := s.sp.Load()
		if sp == nil {
			fail(w, http.StatusServiceUnavailable, "models are still loading")
			return
		}
		h(w, r, sp)
	}
}

func (s *server) models(w http.ResponseWriter, r *http.Request, sp *speech) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	var data []model
	for _, m := range sp.models() {
		data = append(data, model{ID: m, Object: "model", OwnedBy: "hanzo"})
	}
	reply(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// What a caller may ask to be timed.
var granularities = map[string]bool{"word": true, "segment": true}

func (s *server) transcriptions(w http.ResponseWriter, r *http.Request, sp *speech) {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxIn)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		var big *http.MaxBytesError
		if errors.As(err, &big) {
			fail(w, http.StatusRequestEntityTooLarge, "upload exceeds %d bytes", s.maxIn)
			return
		}
		fail(w, http.StatusBadRequest, "expected a multipart form: %v", err)
		return
	}
	defer r.MultipartForm.RemoveAll()
	model := r.FormValue("model")
	if model == "" {
		model = "whisper"
	}
	if sp.ears[model] == nil {
		fail(w, http.StatusNotFound, "unknown model %q; transcription models: %s", model, strings.Join(present(heardBy, sp.ears), ", "))
		return
	}
	lang, err := language(r.FormValue("language"))
	if err != nil {
		fail(w, http.StatusBadRequest, "%v", err)
		return
	}
	format := r.FormValue("response_format")
	if format == "" {
		format = "json"
	}
	if format != "json" && format != "text" && format != "verbose_json" {
		fail(w, http.StatusBadRequest, "unsupported response_format %q; supported: json, text, verbose_json", format)
		return
	}
	// The bracketed name is how a multipart array is spelled: it is what the
	// OpenAI SDKs send and what the ai plane sends. An unrecognized granularity
	// is refused by name; ignoring it would answer 200 with no timings and no
	// reason, which reads as the feature being missing.
	asked := r.MultipartForm.Value["timestamp_granularities[]"]
	var unknown []string
	for _, g := range asked {
		if !granularities[g] {
			unknown = append(unknown, g)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		fail(w, http.StatusBadRequest, "unknown timestamp_granularities %v; supported: segment, word", unknown)
		return
	}
	// Timings ride the verbose body and nowhere else, so asking for them beside a
	// body that cannot carry them is a mistake worth naming.
	if len(asked) > 0 && format != "verbose_json" {
		fail(w, http.StatusBadRequest, "timestamp_granularities requires response_format=verbose_json")
		return
	}
	want := map[string]bool{}
	for _, g := range asked {
		want[g] = true
	}
	if len(want) == 0 {
		want["segment"] = true // OpenAI's default granularity
	}

	f, _, err := r.FormFile("file")
	if err != nil {
		fail(w, http.StatusBadRequest, "a transcription needs a `file`")
		return
	}
	data, err := io.ReadAll(f)
	f.Close()
	if err != nil {
		fail(w, http.StatusBadRequest, "reading the upload: %v", err)
		return
	}
	// max_seconds holds a caller to a length — the ai plane's public lane sends
	// it for a visitor with no account. The decode stops just past it, so the cap
	// bounds the work and not only the answer.
	var longest float64
	if v := r.FormValue("max_seconds"); v != "" {
		longest, err = strconv.ParseFloat(v, 64)
		if err != nil || longest <= 0 {
			fail(w, http.StatusBadRequest, "max_seconds must be a positive number of seconds")
			return
		}
	}
	pcm, err := decode(r.Context(), data, longest)
	if err != nil {
		fail(w, http.StatusBadRequest, "%v", err)
		return
	}
	if longest > 0 && float64(len(pcm)) > (longest+overrun/2)*Rate {
		fail(w, http.StatusRequestEntityTooLarge, "the audio runs past %g s; this request takes at most %g s", longest, longest)
		return
	}
	began := time.Now()
	t, err := sp.transcribe(model, lang, pcm)
	if err != nil {
		fail(w, http.StatusInternalServerError, "transcription failed: %v", err)
		return
	}
	slog.Info("transcribed", "model", model, "language", lang, "audio_s", t.duration, "wall_s", time.Since(began).Seconds())

	switch format {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, t.text)
		return
	case "json":
		// Plain json stays {"text"}, so a client reading the standard shape is
		// unaffected by what verbose_json carries.
		reply(w, http.StatusOK, map[string]any{"text": t.text})
		return
	}
	// verbose_json carries `duration` as OpenAI's does, and the ai plane asks
	// for it because that duration is what meters the call.
	body := map[string]any{"task": "transcribe", "duration": round(t.duration), "text": t.text}
	if want["segment"] {
		body["segments"] = orEmpty(t.segments)
	}
	if want["word"] {
		body["words"] = orEmpty(t.words)
	}
	reply(w, http.StatusOK, body)
}

func orEmpty[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}

func present[T any](names []string, have map[string]T) []string {
	var out []string
	for _, n := range names {
		if _, ok := have[n]; ok {
			out = append(out, n)
		}
	}
	return out
}

// speakRequest is OpenAI's /v1/audio/speech body.
type speakRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	ResponseFormat string   `json:"response_format"`
	Speed          *float64 `json:"speed"`
}

// maxInput is the longest text one request speaks. The ai plane caps callers at
// 4096 characters; this is the same order, for in-cluster callers.
const maxInput = 8192

func (s *server) speech(w http.ResponseWriter, r *http.Request, sp *speech) {
	var ask speakRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&ask); err != nil {
		fail(w, http.StatusBadRequest, "expected a JSON body: %v", err)
		return
	}
	if ask.Model == "" {
		ask.Model = "kokoro"
	}
	m := sp.mouths[ask.Model]
	if m == nil {
		fail(w, http.StatusNotFound, "unknown model %q; speech models: %s", ask.Model, strings.Join(present(spokenBy, sp.mouths), ", "))
		return
	}
	v, err := voice(ask.Model, ask.Voice)
	if err != nil {
		fail(w, http.StatusBadRequest, "%v", err)
		return
	}
	if ask.ResponseFormat == "" {
		ask.ResponseFormat = "mp3" // OpenAI's default, and the one every browser plays
	}
	// Refuse a format we cannot make rather than answering in a different one.
	if _, ok := formats[ask.ResponseFormat]; !ok {
		names := make([]string, 0, len(formats))
		for n := range formats {
			names = append(names, n)
		}
		sort.Strings(names)
		fail(w, http.StatusBadRequest, "unsupported response_format %q; supported: %s", ask.ResponseFormat, strings.Join(names, ", "))
		return
	}
	speed := 1.0
	if ask.Speed != nil {
		speed = *ask.Speed
	}
	if math.IsNaN(speed) || speed < 0.25 || speed > 4.0 {
		fail(w, http.StatusBadRequest, "speed %v is outside 0.25 to 4.0", speed)
		return
	}
	text := strings.TrimSpace(ask.Input)
	if text == "" {
		fail(w, http.StatusBadRequest, "speech needs an `input`")
		return
	}
	if len([]rune(text)) > maxInput {
		fail(w, http.StatusRequestEntityTooLarge, "`input` exceeds %d characters", maxInput)
		return
	}
	began := time.Now()
	said, err := m.speak(text, v, float32(speed))
	if err != nil {
		fail(w, http.StatusInternalServerError, "synthesis failed: %v", err)
		return
	}
	audio, mime, err := encode(r.Context(), said, Spoken, ask.ResponseFormat)
	if err != nil {
		fail(w, http.StatusInternalServerError, "%v", err)
		return
	}
	slog.Info("spoke", "model", ask.Model, "voice", v, "chars", len(text), "audio_s", float64(len(said))/Spoken, "wall_s", time.Since(began).Seconds())
	w.Header().Set("Content-Type", mime)
	_, _ = w.Write(audio)
}

// openRequest opens a growing transcript.
type openRequest struct {
	Model    string `json:"model"`
	Language string `json:"language"`
	Format   string `json:"format"`
	Rate     int    `json:"rate"`
	Channels int    `json:"channels"`
}

func (s *server) transcriptOpen(w http.ResponseWriter, r *http.Request, sp *speech) {
	ask := openRequest{Model: "whisper", Format: "pcm16", Rate: Rate, Channels: 1}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&ask); err != nil && !errors.Is(err, io.EOF) {
		fail(w, http.StatusBadRequest, "expected a JSON body: %v", err)
		return
	}
	if sp.ears[ask.Model] == nil {
		fail(w, http.StatusNotFound, "unknown model %q; transcription models: %s", ask.Model, strings.Join(present(heardBy, sp.ears), ", "))
		return
	}
	// Raw audio carries no header saying what it is, so a mismatch here is not an
	// error downstream: 8 kHz read as 16 kHz transcribes as gibberish. Refuse it
	// by name instead of resampling something we were told is correct.
	if ask.Format != "pcm16" || ask.Rate != Rate || ask.Channels != 1 {
		fail(w, http.StatusBadRequest, "expected pcm16 mono at %d Hz; got %q %dch at %d Hz", Rate, ask.Format, ask.Channels, ask.Rate)
		return
	}
	lang, err := language(ask.Language)
	if err != nil {
		fail(w, http.StatusBadRequest, "%v", err)
		return
	}
	t := newTranscript(sp, ask.Model, lang)
	s.live.begin(t)
	reply(w, http.StatusCreated, map[string]any{
		"id":           t.id,
		"at":           s.here,
		"model":        t.model,
		"format":       "pcm16",
		"rate":         Rate,
		"channels":     1,
		"chunk_ms":     int(math.Round(float64(Chunk) / Second * 1000)),
		"max_bytes":    Ceiling,
		"max_seconds":  Limit,
		"idle_seconds": Idle,
		"expires_at":   time.Now().Add(time.Duration(Idle) * time.Second).Unix(),
	})
}

// transcriptPush takes a chunk and answers with the newest state. `duration` is
// what this push put in front of a decoder.
func (s *server) transcriptPush(w http.ResponseWriter, r *http.Request, sp *speech) {
	id := r.PathValue("id")
	t := s.live.find(id)
	if t == nil {
		fail(w, http.StatusNotFound, "no transcript %q", id)
		return
	}
	pcm, err := io.ReadAll(io.LimitReader(r.Body, Ceiling+1))
	if err != nil {
		fail(w, http.StatusBadRequest, "reading the chunk: %v", err)
		return
	}
	if len(pcm) > Ceiling {
		fail(w, http.StatusRequestEntityTooLarge, "chunk is over %d bytes; limit %d", Ceiling, Ceiling)
		return
	}
	if len(pcm)%Width != 0 {
		fail(w, http.StatusBadRequest, "%d bytes is not whole int16 frames", len(pcm))
		return
	}
	t.mu.Lock()
	full := t.full()
	t.mu.Unlock()
	if full {
		fail(w, http.StatusConflict, "transcript is at its %gs limit; close it", Limit)
		return
	}
	reply(w, http.StatusOK, t.state(t.push(samples(pcm))))
}

// transcriptClose decodes the remainder and commits it. No audio arrives, so
// nothing is billed here: every second was metered by the push that carried it.
func (s *server) transcriptClose(w http.ResponseWriter, r *http.Request, sp *speech) {
	id := r.PathValue("id")
	t := s.live.drop(id)
	if t == nil {
		fail(w, http.StatusNotFound, "no transcript %q", id)
		return
	}
	if err := t.close(); err != nil {
		fail(w, http.StatusInternalServerError, "closing transcript: %v", err)
		return
	}
	reply(w, http.StatusOK, t.state(0))
}

func reply(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// fail answers in OpenAI's error shape, which the ai plane reads the message
// out of (error.message) and hands to the caller.
func fail(w http.ResponseWriter, code int, format string, args ...any) {
	kind := "invalid_request_error"
	if code >= 500 {
		kind = "server_error"
	}
	reply(w, code, map[string]any{"error": map[string]string{"message": fmt.Sprintf(format, args...), "type": kind}})
}

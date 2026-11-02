package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestSpeakerKeepsKokorosLanguagesAndHandsTheRestToSupertonic(t *testing.T) {
	cases := []struct {
		model, lang, text  string
		wantModel, wantLng string
	}{
		{"kokoro", "", "Hello there.", "kokoro", ""},
		{"kokoro", "ja", "こんにちは", "kokoro", ""},
		{"kokoro", "", "مرحباً، كيف حالك اليوم؟", "supertonic", "ar"}, // Arabic script, no language named
		{"kokoro", "ar", "marhaba", "supertonic", "ar"},               // named, whatever the script
		{"kokoro", "de", "Guten Morgen.", "supertonic", "de"},         // no Kokoro voice speaks German
		{"kokoro", "", "Hello مرحبا world and friends", "kokoro", ""}, // mostly Latin stays Kokoro
		{"supertonic", "", "شلونك؟", "supertonic", "ar"},
		{"supertonic", "", "Good morning.", "supertonic", "en"},
		{"supertonic", "ko", "안녕하세요", "supertonic", "ko"},
	}
	for _, c := range cases {
		m, l, err := speaker(c.model, c.lang, c.text)
		if err != nil || m != c.wantModel || l != c.wantLng {
			t.Errorf("speaker(%q, %q, %q) = %q, %q, %v; want %q, %q", c.model, c.lang, c.text, m, l, err, c.wantModel, c.wantLng)
		}
	}
	// Persian is written in Arabic script and neither engine speaks it: refused by name.
	if _, _, err := speaker("kokoro", "fa", "سلام"); err == nil || !strings.Contains(err.Error(), `"fa"`) || !strings.Contains(err.Error(), "ar") {
		t.Fatalf("fa: %v", err)
	}
	if _, _, err := speaker("supertonic", "zh", "你好"); err == nil {
		t.Fatal("supertonic does not speak Mandarin")
	}
}

func TestSupertonicVoicesAndTheVoicesCarriedOver(t *testing.T) {
	for asked, want := range map[string]string{"": "F1", "m3": "M3", "F5": "F5", "nova": "F1", "onyx": "M1", "am_echo": "M1", "bf_emma": "F1"} {
		if got, err := voice("supertonic", asked); err != nil || got != want {
			t.Errorf("voice(supertonic, %q) = %q, %v; want %q", asked, got, err, want)
		}
	}
	if _, err := voice("supertonic", "zz"); err == nil || !strings.Contains(err.Error(), "M5") {
		t.Fatalf("an unknown voice names the ten: %v", err)
	}
	if len(supertonicVoices) != 10 || supertonicID["F1"] != 0 || supertonicID["M1"] != 5 {
		t.Fatal("voice.bin holds F1..F5 then M1..M5")
	}
}

func TestArabicSentToZenVoiceMiniIsSpokenBySupertonic(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	resp, b := r.post("/v1/audio/speech", map[string]any{"model": "kokoro", "voice": "onyx", "input": "مرحباً بك في هانزو", "response_format": "pcm"})
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	if resp.Header.Get("X-Speech-Model") != "supertonic" || resp.Header.Get("Content-Language") != "ar" {
		t.Fatalf("spoken by %q in %q", resp.Header.Get("X-Speech-Model"), resp.Header.Get("Content-Language"))
	}
	if r.tonic.lang != "ar" || r.tonic.voice != "M1" || r.mouth.said != "" {
		t.Fatalf("supertonic got lang %q voice %q; kokoro said %q", r.tonic.lang, r.tonic.voice, r.mouth.said)
	}
	// English stays with kokoro, which says so.
	resp, _ = r.post("/v1/audio/speech", map[string]any{"model": "kokoro", "input": "Welcome to Hanzo.", "response_format": "pcm"})
	if resp.Header.Get("X-Speech-Model") != "kokoro" || r.mouth.said != "Welcome to Hanzo." {
		t.Fatalf("english went to %q", resp.Header.Get("X-Speech-Model"))
	}
	// A language named that no voice speaks is refused, and a bad code is refused by name.
	if resp, b := r.post("/v1/audio/speech", map[string]any{"input": "سلام", "language": "fa"}); resp.StatusCode != 400 || !strings.Contains(string(b), "fa") {
		t.Fatalf("fa: %d %s", resp.StatusCode, b)
	}
	if resp, b := r.post("/v1/audio/speech", map[string]any{"input": "hi", "language": "klingon"}); resp.StatusCode != 400 || !strings.Contains(string(b), "klingon") {
		t.Fatalf("klingon: %d %s", resp.StatusCode, b)
	}
}

// pcm streams: the first sentence reaches the caller before the last is made.
func TestPCMStreamsSentenceBySentence(t *testing.T) {
	r := newRig(t, roomVAD{}, "en")
	slow := &slowMouth{toneMouth: r.tonic, gap: 300 * time.Millisecond}
	r.sp.mouths["supertonic"] = slow
	body, _ := json.Marshal(map[string]any{"model": "supertonic", "input": "مرحباً. كيف حالك. أنا بخير.", "response_format": "pcm"})
	began := time.Now()
	resp, err := http.Post(r.srv.URL+"/v1/audio/speech", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	rd := bufio.NewReader(resp.Body)
	if _, err := rd.Peek(2); err != nil {
		t.Fatal(err)
	}
	first := time.Since(began)
	n := 0
	for {
		b, err := rd.ReadByte()
		if err != nil {
			break
		}
		_ = b
		n++
	}
	all := time.Since(began)
	if first > all/2 || all < 3*slow.gap {
		t.Fatalf("first audio after %v, all after %v: the stream waited for the whole reply", first, all)
	}
	if n%2 != 0 || n == 0 {
		t.Fatalf("%d bytes is not pcm16", n)
	}
}

type slowMouth struct {
	*toneMouth
	gap time.Duration
}

func (m *slowMouth) speak(text, voice, lang string, speed float32) ([]float32, error) {
	time.Sleep(m.gap) // each sentence takes this long to make
	return m.toneMouth.speak(text, voice, lang, speed)
}

// A growing transcript says which language it heard, once it knows.
func TestTranscriptReportsTheLanguageItIdentified(t *testing.T) {
	r := newRig(t, alwaysVAD{}, "ar")
	resp, b := r.post("/v1/audio/transcript", map[string]any{"model": "parakeet", "format": "pcm16", "rate": 16000, "channels": 1})
	if resp.StatusCode != 201 {
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	var open struct{ ID string }
	_ = json.Unmarshal(b, &open)
	tone := make([]byte, 4*Second)
	for i := 0; i+1 < len(tone); i += 2 {
		tone[i] = byte(i)
	}
	var st state
	for i := 0; i < 4; i++ {
		_, b = r.post("/v1/audio/transcript/"+open.ID, tone[i*Second:(i+1)*Second])
		_ = json.Unmarshal(b, &st)
	}
	time.Sleep(200 * time.Millisecond)
	_, b = r.delete("/v1/audio/transcript/" + open.ID)
	_ = json.Unmarshal(b, &st)
	if st.Language != "ar" {
		t.Fatalf("language %q after identifying Arabic: %s", st.Language, b)
	}
	// named up front, it is reported at once
	_, b = r.post("/v1/audio/transcript", map[string]any{"model": "whisper", "language": "ar-EG"})
	_ = json.Unmarshal(b, &open)
	_, b = r.post("/v1/audio/transcript/"+open.ID, tone[:Second])
	_ = json.Unmarshal(b, &st)
	if st.Language != "ar" {
		t.Fatalf("named ar-EG, reported %q", st.Language)
	}
}

func TestSentencesOfCutsAtLatinAndArabicMarks(t *testing.T) {
	got := sentencesOf("مرحباً. كيف حالك؟ أنا بخير! Fine.\nOK 3.5 it is")
	want := []string{"مرحباً.", "كيف حالك؟", "أنا بخير!", "Fine.", "OK 3.5 it is"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("%q", got)
	}
}

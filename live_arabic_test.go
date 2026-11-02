package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode"
)

// Arabic, end to end on the real weights: zen-voice-mini's upstream (kokoro) is
// handed Arabic text, supertonic says it, and whisper must say it back. The
// lines are Modern Standard, Gulf and Egyptian; whisper writes all three in its
// own spelling, so the bound is a character error rate after folding the
// letters Arabic spells more than one way. Measured on dgx at eight steps: 0, 0 and 8.8%
// ("قصدك" heard as "الصدق"); the bound sits above that and far below a wrong
// language or a garbled voice, which score near 1.
func TestLiveArabicIsSpokenBySupertonicAndHeardBack(t *testing.T) {
	sp := live(t)
	r := &rig{t: t, s: &server{maxIn: 100 << 20}}
	r.s.sp.Store(sp)
	r.srv = httptest.NewServer(r.s.routes())
	defer r.srv.Close()

	for _, line := range []string{
		"تقع مدينة الرباط على ساحل المحيط الأطلسي، وهي عاصمة المملكة المغربية.", // Modern Standard
		"هلا والله، شلونك اليوم؟ وش تبي أسوي لك؟",                               // Gulf
		"أنا مش فاهم قصدك، ممكن تقولها تاني بالراحة؟",                           // Egyptian
	} {
		began := time.Now()
		resp, b := r.post("/v1/audio/speech", map[string]any{"model": "kokoro", "input": line, "response_format": "wav"})
		took := time.Since(began)
		if resp.StatusCode != 200 || resp.Header.Get("X-Speech-Model") != "supertonic" || resp.Header.Get("Content-Language") != "ar" {
			t.Fatalf("speech: %d %s %q %q", resp.StatusCode, b[:min(80, len(b))], resp.Header.Get("X-Speech-Model"), resp.Header.Get("Content-Language"))
		}
		secs := float64(len(b)-44) / 2 / Spoken
		resp, b = r.transcribe(b, map[string][]string{"model": {"whisper"}, "language": {"ar"}, "response_format": {"json"}})
		if resp.StatusCode != 200 {
			t.Fatalf("transcribe: %d %s", resp.StatusCode, b)
		}
		var heard struct{ Text string }
		_ = json.Unmarshal(b, &heard)
		e := cer(line, heard.Text)
		t.Logf("%.2f s of audio in %.2f s (RTF %.3f): %q -> %q, CER %.3f", secs, took.Seconds(), took.Seconds()/secs, line, heard.Text, e)
		if e > 0.15 {
			t.Fatalf("character error %.3f: said %q, heard %q", e, line, heard.Text)
		}
	}
}

// Through the handler, a pcm stream's first sentence arrives before its last
// is made: the time to the first byte is well under the time to the end.
func TestLiveArabicPCMStreams(t *testing.T) {
	sp := live(t)
	r := &rig{t: t, s: &server{maxIn: 100 << 20}}
	r.s.sp.Store(sp)
	r.srv = httptest.NewServer(r.s.routes())
	defer r.srv.Close()
	body, _ := json.Marshal(map[string]any{"model": "supertonic", "voice": "M1", "response_format": "pcm",
		"input": "مرحباً. أنا مساعدك الصوتي من هانزو، وأستطيع أن أجيب عن أسئلتك بالعربية. كيف يمكنني مساعدتك اليوم؟"})
	began := time.Now()
	resp, err := http.Post(r.srv.URL+"/v1/audio/speech", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 2)
	if _, err := io.ReadFull(resp.Body, buf); err != nil {
		t.Fatal(err)
	}
	first := time.Since(began)
	rest, _ := io.ReadAll(resp.Body)
	all := time.Since(began)
	secs := float64(len(rest)+2) / 2 / Spoken
	t.Logf("first audio after %.2f s, %.2f s of audio after %.2f s", first.Seconds(), secs, all.Seconds())
	if first > all/2 {
		t.Fatalf("first audio after %v of %v: the stream waited for the whole reply", first, all)
	}
}

// cer is the character error rate of got against want, Arabic folded: no
// diacritics or tatweel, the hamza seats and alef forms as bare alef, alef
// maqsura as ya, ta marbuta as ha, no punctuation and no spaces.
func cer(want, got string) float64 {
	fold := strings.NewReplacer("أ", "ا", "إ", "ا", "آ", "ا", "ٱ", "ا", "ى", "ي", "ة", "ه", "ؤ", "و", "ئ", "ي")
	norm := func(s string) []rune {
		var out []rune
		for _, r := range fold.Replace(s) {
			if unicode.Is(unicode.Mn, r) || r == 'ـ' || !(unicode.IsLetter(r) || unicode.IsDigit(r)) {
				continue
			}
			out = append(out, unicode.ToLower(r))
		}
		return out
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

package main

import (
	"math"
	"testing"
)

// A tone keeps its pitch and its level across 44.1 kHz -> 24 kHz, and a tone
// above the new Nyquist is filtered out rather than folded back in.
func TestResampleKeepsToneAndDropsWhatCannotBeCarried(t *testing.T) {
	tone := func(hz float64, rate, n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(0.5 * math.Sin(2*math.Pi*hz*float64(i)/float64(rate)))
		}
		return x
	}
	in := tone(1000, 44100, 44100)
	out := resample(in, 44100, Spoken)
	if len(out) != Spoken {
		t.Fatalf("one second at 44.1 kHz became %d samples, want %d", len(out), Spoken)
	}
	// compare against the ideal 24 kHz tone away from the edges
	ref := tone(1000, Spoken, Spoken)
	var err, pow float64
	for i := 200; i < len(out)-200; i++ {
		d := float64(out[i] - ref[i])
		err += d * d
		pow += float64(ref[i]) * float64(ref[i])
	}
	if snr := 10 * math.Log10(pow/err); snr < 40 {
		t.Fatalf("1 kHz tone resampled at %.1f dB SNR, want at least 40", snr)
	}
	alias := resample(tone(15000, 44100, 44100), 44100, Spoken) // 15 kHz is above 12 kHz Nyquist
	var peak float64
	for i := 200; i < len(alias)-200; i++ {
		peak = math.Max(peak, math.Abs(float64(alias[i])))
	}
	if peak > 0.01 {
		t.Fatalf("a 15 kHz tone came through at %.3f full scale; it would alias to 9 kHz", peak)
	}
	if got := resample(in, Spoken, Spoken); &got[0] != &in[0] {
		t.Fatal("same rate should return the input untouched")
	}
}

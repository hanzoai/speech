package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
)

// decode turns any container a browser records — webm/opus from Chrome and
// Firefox, ogg, mp4/m4a from Safari, wav, mp3 — into 16 kHz mono samples.
//
// ffmpeg reads a FILE, not a pipe: an mp4 whose index (moov) sits after its
// data cannot be demuxed from a stream that cannot seek, and that is the layout
// most m4a files have. The container is sniffed from the bytes, never from the
// filename — the ai plane names every upload audio.webm whatever it is.
//
// longest, when positive, stops the decode just past that many seconds, so a
// caller held to a length pays for no more decoding than the length allows: a
// file far longer than the cap costs a fraction of a second, not its whole run.
func decode(ctx context.Context, data []byte, longest float64) ([]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("the file is empty")
	}
	f, err := os.CreateTemp("", "speech-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(data); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	var out, errs bytes.Buffer
	args := []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-i", f.Name(), "-vn", "-ac", "1", "-ar", fmt.Sprint(Rate)}
	if longest > 0 {
		args = append(args, "-t", fmt.Sprintf("%.3f", longest+overrun))
	}
	cmd := exec.CommandContext(ctx, "ffmpeg", append(args, "-f", "f32le", "pipe:1")...)
	cmd.Stdout, cmd.Stderr = &out, &errs
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(errs.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("not audio ffmpeg can read: %s", lastLine(msg))
	}
	b := out.Bytes()
	pcm := make([]float32, len(b)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return pcm, nil
}

// overrun is how far past a cap the decode reads, so audio that runs over is told
// from audio that ends exactly at it.
const overrun = 0.25

func lastLine(s string) string {
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return s[i+1:]
	}
	return s
}

// samples reads raw little-endian int16 as the float in [-1, 1) a model takes.
func samples(pcm []byte) []float32 {
	out := make([]float32, len(pcm)/Width)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
	}
	return out
}

// pcm16 writes samples as little-endian int16, clipped to full scale.
func pcm16(s []float32) []byte {
	out := make([]byte, 2*len(s))
	for i, x := range s {
		v := math.Round(float64(x) * 32767)
		v = math.Max(-32768, math.Min(32767, v))
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v)))
	}
	return out
}

// wav wraps samples as a 16-bit mono RIFF/WAVE file at rate.
func wav(s []float32, rate int) []byte {
	data := pcm16(s)
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	w(uint32(36 + len(data)))
	b.WriteString("WAVEfmt ")
	w(uint32(16))
	w(uint16(1)) // PCM
	w(uint16(1)) // mono
	w(uint32(rate))
	w(uint32(rate * 2))
	w(uint16(2))
	w(uint16(16))
	b.WriteString("data")
	w(uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}

// format is a response container: the media type it really is, and the ffmpeg
// encode that makes it from wav. wav and pcm need no encode — they are written
// here, so asking for raw samples never spawns a process to strip a header.
//
// opus is Ogg-contained, so it is audio/ogg: that is what the bytes are. A
// caller handed a container other than the one the label names is entitled to
// refuse it, so every format here is really encoded, and anything else is
// refused by name rather than answered in a different one.
type format struct {
	mime   string
	encode []string
}

var formats = map[string]format{
	"mp3":  {"audio/mpeg", []string{"-f", "mp3", "-c:a", "libmp3lame"}},
	"wav":  {"audio/wav", nil},
	"opus": {"audio/ogg", []string{"-f", "ogg", "-c:a", "libopus"}},
	"aac":  {"audio/aac", []string{"-f", "adts", "-c:a", "aac"}},
	"flac": {"audio/flac", []string{"-f", "flac"}},
	"pcm":  {"audio/pcm", nil},
}

// encode renders synthesized samples in a format, returning the bytes and the
// media type they are.
func encode(ctx context.Context, s []float32, rate int, name string) ([]byte, string, error) {
	f, ok := formats[name]
	if !ok {
		return nil, "", fmt.Errorf("unsupported response_format %q", name)
	}
	switch name {
	case "pcm":
		return pcm16(s), f.mime, nil
	case "wav":
		return wav(s, rate), f.mime, nil
	}
	args := append([]string{"-nostdin", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"}, f.encode...)
	cmd := exec.CommandContext(ctx, "ffmpeg", append(args, "pipe:1")...)
	var out, errs bytes.Buffer
	cmd.Stdin, cmd.Stdout, cmd.Stderr = bytes.NewReader(wav(s, rate)), &out, &errs
	if err := cmd.Run(); err != nil {
		return nil, "", fmt.Errorf("encode %s: %s", name, lastLine(strings.TrimSpace(errs.String()+" "+err.Error())))
	}
	return out.Bytes(), f.mime, nil
}

// resample converts mono samples from one rate to another with a windowed-sinc
// filter: a Blackman window over taps input samples either side of each output
// instant, the cutoff at 0.95 of the lower Nyquist so a downsample does not
// fold what it cannot carry back into what it keeps. The ratio is reduced to
// lowest terms and the filter for each of its phases is built once, so a
// 44.1 kHz to 24 kHz pass (147:80) evaluates 80 filters, never a sine per tap.
func resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	g := gcd(from, to)
	up, down := to/g, from/g // output n sits at input n*down/up
	cut := 0.95 * 0.5 * math.Min(1, float64(to)/float64(from))
	const taps = 16
	reach := int(math.Ceil(taps / (2 * cut) / 2)) // input samples either side, wider when cutting lower
	width := 2*reach + 1
	bank := make([][]float32, up)
	for p := range bank {
		frac := float64(p) / float64(up)
		h := make([]float32, width)
		var sum float64
		for j := range h {
			d := float64(j-reach) - frac // distance from the output instant, in input samples
			w := 0.42 + 0.5*math.Cos(math.Pi*d/float64(reach+1)) + 0.08*math.Cos(2*math.Pi*d/float64(reach+1))
			if math.Abs(d) > float64(reach+1) {
				w = 0
			}
			s := 2 * cut
			if d != 0 {
				s = math.Sin(2*math.Pi*cut*d) / (math.Pi * d)
			}
			h[j] = float32(s * w)
			sum += s * w
		}
		for j := range h { // unity gain at DC, whatever the phase
			h[j] = float32(float64(h[j]) / sum)
		}
		bank[p] = h
	}
	n := len(x) * up / down
	out := make([]float32, n)
	for i := range out {
		pos := i * down
		at, p := pos/up, pos%up
		h := bank[p]
		var acc float32
		for j, c := range h {
			k := at + j - reach
			if k >= 0 && k < len(x) {
				acc += x[k] * c
			}
		}
		out[i] = acc
	}
	return out
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

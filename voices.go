package main

import (
	"fmt"
	"sort"
	"strings"
)

// kokoroVoices are Kokoro v1.0's voices, in speaker-id order: the index IS the
// id the model takes. The order is the model's own metadata (speaker_names);
// newKokoro checks the count against the loaded model, so a different voice
// table fails at boot rather than speaking in the wrong voice.
var kokoroVoices = []string{
	"af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
	"af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
	"am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
	"bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "ff_siwis", "hf_alpha",
	"hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi",
	"jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao",
	"zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang", "em_santa",
}

var kokoroID = func() map[string]int {
	m := make(map[string]int, len(kokoroVoices))
	for i, v := range kokoroVoices {
		m[v] = i
	}
	return m
}()

// openaiVoices lets an OpenAI SDK caller keep its voice names. Where Kokoro has a
// voice of the same name it is that voice; the others go to the best-rated
// Kokoro voice of the same character.
var openaiVoices = map[string]string{
	"alloy":   "af_alloy",
	"ash":     "am_michael",
	"ballad":  "bm_george",
	"coral":   "af_sarah",
	"echo":    "am_echo",
	"fable":   "bm_fable",
	"nova":    "af_nova",
	"onyx":    "am_onyx",
	"sage":    "af_kore",
	"shimmer": "af_bella",
	"verse":   "am_puck",
}

// kokoroLang is the language a Kokoro voice speaks, from the first letter of its
// id, as the name of an espeak-ng voice FILE: sherpa hands it to
// espeak_SetVoiceByName, which matches file names, so "en-gb" and "fr-fr" — the
// codes Kokoro's own pipeline uses — find nothing and the request fails to
// tokenize. "en" is espeak's British English and "fr" its French. Mandarin text
// is read through the Chinese lexicon whatever this says; "cmn" is for what is
// not Han — digits, mostly.
func kokoroLang(voice string) string {
	switch voice[0] {
	case 'b':
		return "en"
	case 'e':
		return "es"
	case 'f':
		return "fr"
	case 'h':
		return "hi"
	case 'i':
		return "it"
	case 'j':
		return "ja"
	case 'p':
		return "pt-br"
	case 'z':
		return "cmn"
	}
	return "en-us"
}

// supertonicVoices are Supertonic 3's ten voice styles in speaker-id order: the
// index is the sid sherpa takes. voice.bin holds F1..F5 then M1..M5 (measured
// against the published voice_styles/*.json); newSupertonic checks the count.
var supertonicVoices = []string{"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"}

var supertonicID = func() map[string]int {
	m := make(map[string]int, len(supertonicVoices))
	for i, v := range supertonicVoices {
		m[v] = i
	}
	return m
}()

func supertonicHas(v string) bool { _, ok := supertonicID[v]; return ok }

// defaultVoice is what a model speaks in when the caller names no voice.
var defaultVoice = map[string]string{"kokoro": "af_heart", "supertonic": "F1"}

// carried is the supertonic voice that stands in for a kokoro or OpenAI voice
// when a request moves to supertonic for its language: the same character —
// a woman's voice for a woman's, a man's for a man's — so a caller who chose
// "nova" or "onyx" and then asked for Arabic still hears the voice they chose.
func carried(asked string) (string, bool) {
	k, ok := openaiVoices[asked]
	if !ok {
		k = asked
	}
	if _, ok := kokoroID[k]; !ok || len(k) < 2 {
		return "", false
	}
	if k[1] == 'm' {
		return "M1", true
	}
	return "F1", true
}

// voice resolves a caller's voice for a model, or says which ones exist.
func voice(model, asked string) (string, error) {
	v := strings.ToLower(strings.TrimSpace(asked))
	if v == "" {
		return defaultVoice[model], nil
	}
	switch model {
	case "kokoro":
		if id, ok := openaiVoices[v]; ok {
			return id, nil
		}
		if _, ok := kokoroID[v]; ok {
			return v, nil
		}
		names := make([]string, 0, len(openaiVoices))
		for n := range openaiVoices {
			names = append(names, n)
		}
		sort.Strings(names)
		return "", fmt.Errorf("unknown voice %q for kokoro; voices: %s; OpenAI names: %s",
			asked, strings.Join(kokoroVoices, ", "), strings.Join(names, ", "))
	case "supertonic":
		if u := strings.ToUpper(v); supertonicHas(u) {
			return u, nil
		}
		if c, ok := carried(v); ok {
			return c, nil
		}
		return "", fmt.Errorf("unknown voice %q for supertonic; voices: %s, or a kokoro or OpenAI voice name",
			asked, strings.Join(supertonicVoices, ", "))
	}
	return "", fmt.Errorf("model %q speaks nothing", model)
}

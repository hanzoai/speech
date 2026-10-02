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

// defaultVoice is what a model speaks in when the caller names no voice.
var defaultVoice = map[string]string{"kokoro": "af_heart"}

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
	}
	return "", fmt.Errorf("model %q speaks nothing", model)
}
